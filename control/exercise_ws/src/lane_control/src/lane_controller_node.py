#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Segment, SegmentList,Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.pp_controller = PurePursuitLaneController(la_dist=0.2, d=0, phi=0, v_ref=0.22)
        #sim PurePursuitLaneController(la_dist=0.20, d=0, phi=0, gain = 0.3, v_ref=0.3)
        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        self.sub_lineseglist = rospy.Subscriber("~/agent/lane_filter_node/seglist_filtered",
                                                SegmentList,
                                                self.cbLineSeg,
                                                queue_size=1)
        
        

        self.log("Initialized!")


    def cbLineSeg(self, input_segments):
        #print(input_segments.segments)
        self.pp_controller.segments = input_segments.segments  #self.pp_controller.get_segments(input_segments.segments) #input_segments.segments#
    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """      
        
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        self.cbParametersChanged()
        car_control_msg.v, car_control_msg.omega = self.pp_controller.calculate_speed() #get_angular_velocity(self.pp_controller.la)
        if self.pp_controller.slow:
            car_control_msg.v = 0.7 * car_control_msg.v
        #print(car_control_msg.omega)
        self.publishCmd(car_control_msg)


    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        
        self.pub_car_cmd.publish(car_cmd_msg)
        
    def cbParametersChanged(self):
    #    """Updates parameters in the controller object."""
        #print(self.pose_msg.phi)
        self.pp_controller.update_parameters(self.pose_msg.d, self.pose_msg.phi)

if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
