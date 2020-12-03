import os
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
import rospy


class LaneFilterHistogramKF:
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            "mean_d_0",
            "mean_phi_0",
            "sigma_d_0",
            "sigma_phi_0",
            "delta_d",
            "delta_phi",
            "d_max",
            "d_min",
            "phi_max",
            "phi_min",
            "cov_v",
            "linewidth_white",
            "linewidth_yellow",
            "lanewidth",
            "min_max",
            "sigma_d_mask",
            "sigma_phi_mask",
            "range_min",
            "range_est",
            "range_max",
            "wheel_radius",
            "baseline",
            "sigma_q_d",  # process noise d
            "sigma_q_phi",  # process noise phi
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.belief = {"mean": self.mean_0, "covariance": self.cov_0}

        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.baseline = 0.0
        self.initialized = False
        self.reset()

    def reset(self):
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.belief = {"mean": self.mean_0, "covariance": self.cov_0}

        # d_resolution: 0.011 for ref from controller
        # phi_resolution: 0.051

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        # TODO update self.belief based on right and left encoder data + kinematics
        if not self.initialized:
            return
        # process noise
        Q = np.array([[self.sigma_q_d, 0], [0, self.sigma_q_phi]])
        # distance traveled by wheels
        dist_r = (
            right_encoder_delta
            / self.encoder_resolution
            * 2
            * np.pi
            * self.wheel_radius
        )
        dist_l = (
            left_encoder_delta / self.encoder_resolution * 2 * np.pi * self.wheel_radius
        )
        # Change in phi from distance traveled by the wheels and the distance between the wheels

        delta_phi = (
            dist_r - dist_l
        ) / self.baseline  # we actually would need to multiply by dt is using v so using distance
        mu_phi = self.belief["mean"][1] + delta_phi
        delta_d = np.sin(mu_phi) * (dist_l + dist_r) / 2

        mu_prev = self.belief["mean"]  # [mu_d, mu_phi]
        # cov_prev = self.belief["covariance"]
        mu_pred = [mu_prev[0] + delta_d, mu_phi]
        cov_pred = Q
        # print("predicted mean", predicted_mu)
        # print("predicted cov", predicted_cov)
        self.belief["mean"] = mu_pred
        self.belief["covariance"] = self.belief["covariance"] + cov_pred

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays

        measurement_likelihood = self.generate_measurement_likelihood(segmentsArray)
        if measurement_likelihood is None:
            print("no measurement likelihood")
            return ()
        # TODO: Parameterize the measurement likelihood as a Gaussian
        cov_prev = self.belief["covariance"]
        H = np.eye(2)
        # print(measurement_likelihood.shape)
        # get position of cell with max value
        i_max, j_max = np.unravel_index(
            np.argmax(measurement_likelihood, axis=None), measurement_likelihood.shape
        )
        # get d, phi where measurement likelihood is at max
        # The predicted measurement is basically the predicted state.
        H = np.array([[1, 0], [0, 1]])

        # observation noise
        # compute the mean, variance given histogram
        d = np.arange(self.d_min, self.d_max, self.delta_d)
        phi = np.arange(self.phi_min, self.phi_max, self.delta_phi)

        margin_d = measurement_likelihood.sum(axis=1)
        margin_phi = measurement_likelihood.sum(axis=0)
        d_mean = (
            self.d_min + (i_max + 0.5) * self.delta_d
        )  # np.multiply(d, margin_d).sum()
        phi_mean = self.phi_min + (j_max + 0.5) * self.delta_phi  # this takes max
        # np.multiply(phi, margin_phi).sum() #this takes expectation
        pred_state = np.array([d_mean, phi_mean])
        cov_d_phi = np.multiply(
            np.outer(d - d_mean, np.transpose(phi - phi_mean)), measurement_likelihood
        ).sum()

        var_d = (
            np.sum(np.multiply((d - d_mean) ** 2, margin_d))
            * len(margin_d)
            / (len(margin_d) - 1)
        )
        var_phi = (
            np.sum(np.multiply((phi - phi_mean) ** 2, margin_phi))
            * len(margin_phi)
            / (len(margin_phi) - 1)
        )
        R = np.array([[var_d, cov_d_phi], [cov_d_phi, var_phi]])

        # TODO: Apply the update equations for the Kalman Filter to self.belief
        residual_cov = H @ cov_prev @ H.T + R
        try:
            K = cov_prev @ H.T @ np.linalg.inv(residual_cov)
        except np.linalg.LinAlgError:
            K = np.zeros((2, 2))

        meas_diff = pred_state - np.dot(H, self.belief["mean"])
        self.belief["mean"] = self.belief["mean"] + np.dot(K, meas_diff)
        self.belief["covariance"] = cov_prev - K @ H @ cov_prev

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[
            self.d_min : self.d_max : self.delta_d,
            self.phi_min : self.phi_max : self.delta_phi,
        ]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            # print(segment.color)
            d_i, phi_i, l_i, weight = self.generateVote(segment)
            # print(d_i, phi_i)
            # if the vote lands outside of the histogram discard it
            if (
                d_i > self.d_max
                or d_i < self.d_min
                or phi_i < self.phi_min
                or phi_i > self.phi_max
            ):
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / np.sum(measurement_likelihood)
        return measurement_likelihood

    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if l1 < 0:
            l1 = -l1
        if l2 < 0:
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if p1[0] > p2[0]:  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = -d_i  # -d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2
        elif segment.color == segment.YELLOW:  # left lane is yellow
            if p2[0] > p1[0]:  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i  # -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if (
                abs(d_s - d_max) < 3 * self.delta_d
                and abs(phi_s - phi_max) < 3 * self.delta_phi
            ):
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c ** 2 + y_c ** 2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray
