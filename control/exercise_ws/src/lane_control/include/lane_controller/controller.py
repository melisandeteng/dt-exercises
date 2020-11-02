import numpy as np
class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.
    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """
    def __init__(self, la_dist, d, phi, v_ref=0.5):
        """[summary]

        Args:
            la_dist ([type]): lookahead distance - max distance to keep considering segments
            d ([type]): distance to center of the lane from lane pose
            phi ([type]): angle from center lane from lane pose
            gain ([type]): [description]
            tolerance (float, optional): [description]. Defaults to 0.1.
        """
        self.la_dist = la_dist
        self.d_est = d
        self.phi_est = phi
        self.colors = {0, 1} #only consider yellow and white detected lines 
        self.lane_width = 0.25
        self.v_ref = v_ref
        self.segments = [] #segmentss to be considered for computation of lane center
        self.thresh_angle = 10
        self.tol = 0.05    
        self.max_dist = 0.35
        self.slow = False
        self.max_seg = 6
  
# ---------------------------
# ----- GET LANE CENTER -----
# ---------------------------

    def get_lc_from_lane_pose(self, dist):
        """get lane center at dist away from closest lanne center point to the robot,
         as though that there was no turn in the lane section, using lane pose info
        """
        T_or = np.array([[np.cos(self.phi_est), -np.sin(self.phi_est),0],
                          [np.sin(self.phi_est), np.cos(self.phi_est),  self.d_est],
                          [0, 0, 1]
                         ]) # Transformation matrix from origin to robot
        T_of = np.array([
            [1, 0, dist],
            [0, 1, 0],
            [0, 0, 1]])
        T_rf = np.dot(np.linalg.inv(T_or), T_of) # Transformation matrix from robot to follow point
        return([T_rf[0,2], T_rf[1,2]])

    def get_lc_from_seg(self, segment):
        """Get point at center of the lane given one segment 
        Args:
            segment ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Compute normal vector to the segment
        
        segment.points = sorted(segment.points, key=lambda point: point.x)
        
        point_1 = segment.points[0]
        point_2 = segment.points[1]
        dy, dx = (point_2.y - point_1.y), point_2.x - point_1.x
        norm = np.sqrt(dx ** 2 + dy ** 2) 
        # assume yellow is always on left, white on right
        if segment.color == 0:  #white
            x, y = - dy / norm, dx / norm
            if np.abs(x) > np.cos(np.deg2rad(self.thresh_angle)):
                return([0, 0.25])
        elif segment.color == 1:
            x, y = dy / norm, -dx / norm
            if np.abs(x) > np.cos(np.deg2rad(self.thresh_angle)):
                return([0, -0.25])
        else:
            x, y = 0,0
      
        lane_center_x = point_1.x + x * (self.lane_width / 2)
        lane_center_y = point_1.y + y * (self.lane_width / 2)
        return ([lane_center_x, lane_center_y])
    
    def get_lookahead(self, segment_list, dist):
        """[summary]

        Args:
            yellow_segs ([type]): [description]
            white_segs ([type]): [description]

        Returns:
            [type]: [description]
        """
        if len(segment_list)==0:
            self.slow = True
            return(self.get_lc_from_lane_pose(dist))#, self.v_ref, 1)
        else: 
            coords = []
            dists = []
            for s in segment_list :
                x, y = self.get_lc_from_seg(s)
                #if x!=100:
                if self.get_distance_point(x, y) < self.max_dist:
                    coords += [(x,y)]
                #dists = []
                #for c in coords:
                    dists += [self.get_distance_point(x, y)]
            if len(coords)==0:
                self.slow = True
                return(self.get_lc_from_lane_pose(dist))#self.get_lc_from_lane_pose(dist))
          
            dists = [(d - self.la_dist) for d in dists]
            indices = np.argsort(dists)
            keep = [coords[i] for i in indices[:min(len(coords),self.max_seg)]]
            self.slow = False
            return (np.mean([coord[0] for coord in keep]), np.mean([coord[1] for coord in keep]))
            #else: 
            #    print("emergency")
            #    return (-100,y)

# -----------------
# ----- UTILS -----
# -----------------
    def get_distance_point(self, coord1, coord2):
        point_1 = coord1
        point_2 = coord2
        #mid_x, mid_y = (point_2 + point_1)/2, (point_2.y + point_1.y)/2
        return (np.sqrt(point_1**2+ point_2**2))
    
    def filter_seg(self, segments, d_min, d_max):
        """[summary]

        Args:
            segments (SegmentList): [description]
            d_min ([type]): [description]
            d_max ([type]): [description]
        """
        keep = []
        dists = []
        for segment in segments:
            segment.points = sorted(segment.points, key=lambda point: point.x)
            dist = self.get_distance_seg(segment) 
            if segment.color in self.colors and dist < d_max and dist > d_min:
                keep += [segment]
                dists += [np.abs(dist - self.la_dist)]
        #sort by closest to furthest from lookahead
        indices = np.argsort(dists)
        keep = [keep[i] for i in indices]
        
        return(keep)

    def get_segments(self, segment_list):
        segs = self.filter_seg(segment_list, self.d_min, self.d_max)
        if len(segs)>=1:
            segs = segs[:min(self.max_seg, len(segs))]
        self.segments = segs
        


# ------------------------
# ----- GET V, OMEGA -----
# ------------------------

    def emergency_turn(self, normvec_x_dir,  color, angle=10):
        # Compute normal vector to the segment
        v_mult = 0
        if np.abs(normvec_x_dir) > np.cos(np.deg2rad(angle)):
            if color == 0:#white
                v_mult = -1
            if color == 1:
                v_mult = 1
        return(v_mult)
            

    def calculate_speed(self):
        la = self.get_lookahead(self.segments, self.la_dist)
    
        #if la[0]==-100 :
        #    print("try to turn")
        #    return(0.1, la[1]*3)
        omega = self.get_angular_velocity(la)
        return(self.v_ref, omega)

    def get_angular_velocity(self, la_coords):
        """
        Return the angular velocity in order to control the Duckiebot so that it follows the lane.
        Parameters:2
             d_est: distance from the center of the lane that we get from the lane pose
             phi_est: angle from the lane direction, that we get from the lane pose
        Returns:
            omega: angular velocity, in rad/sec. 
        """
        x_la, y_la = la_coords
        L =  np.sqrt(x_la** 2 + y_la ** 2)
        sin_alpha = y_la / np.sqrt(x_la** 2 + y_la ** 2)
        omega = sin_alpha *1.5*self.v_ref/ L #sin_alpha *1.5*self.v_ref/ L for sim

        return omega
    
    def update_parameters(self, d, phi):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.d_est = d
        self.phi_est = phi
        
