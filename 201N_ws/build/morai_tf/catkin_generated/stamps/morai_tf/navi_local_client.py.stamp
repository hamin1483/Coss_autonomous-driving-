#! /usr/bin/env python3

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped

class NavigationClient():
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.goal_status_sub = rospy.Subscriber('/move_base/status', GoalStatus, self.goal_status_CB)
        self.client.wait_for_server()

        self.mission_flag_pub = rospy.Publisher('/mission1', Bool, queue_size=10)
        self.flag_msg = Bool()
        self.flag = False
        
        self.param_name = '~flag_param'
        rospy.set_param(self.param_name, self.flag)
        
        self.goal_list = []  # 목표 지점 리스트

        self.waypoint_1 = MoveBaseGoal()  # 도착 지점
        self.waypoint_1.target_pose.header.frame_id = 'map'
        self.waypoint_1.target_pose.pose.position.x = 16.6145030387823
        self.waypoint_1.target_pose.pose.position.y = -9.891198457135985
        self.waypoint_1.target_pose.pose.orientation.z = -0.007985089933669315
        self.waypoint_1.target_pose.pose.orientation.w = 0.9999681186611658
        
        self.goal_list.append(self.waypoint_1)

        self.sequence = 0
        self.start_time = rospy.Time.now()

        # Publisher for /initialpose
        self.initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    def goal_status_CB(self, msg):
        # Check the status_list for the desired condition
        for status in msg.status_list:
            if status.status == 3:  # Check for the status code you want (3 for SUCCEEDED)
                self.flag = True
                rospy.loginfo("Goal succeeded. Setting self.flag to True.")
    def run(self):
        if not self.flag and self.client.get_state() != GoalStatus.ACTIVE:
            # Publish initial pose before sending move_base goal
            self.publish_initial_pose()
            self.start_time = rospy.Time.now()
            self.client.send_goal(self.goal_list[self.sequence])
                
        elif self.flag == True:
            rospy.loginfo("Goal Reached")
            self.client.cancel_all_goals()
            self.flag_msg.data = self.flag
            self.mission_flag_pub.publish(self.flag_msg)
            rospy.set_param(self.param_name, True)  # 플래그 값을 파라미터 서버에 저장

    def publish_initial_pose(self):
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.frame_id = 'map'

        # Set your desired initial pose values here
        initial_pose_msg.pose.pose.position.x = 0.060722974402695015
        initial_pose_msg.pose.pose.position.y = 0.0010099079608249146
        initial_pose_msg.pose.pose.orientation.z = 0.03350902978386785
        initial_pose_msg.pose.pose.orientation.w = 0.9994384147724881

        self.initialpose_pub.publish(initial_pose_msg)

def main():
    rospy.init_node('navigation_client')
    nc = NavigationClient()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown() and not nc.flag:
        nc.run()
        if nc.flag:  # flag가 True일 때 루프를 빠져나옴
            break
        rate.sleep()
    
    while not rospy.is_shutdown() and nc.flag:
        nc.mission_flag_pub.publish(True)

if __name__ == "__main__":
    main()


# position: 
#       x: 18.116541588572197
#       y: -9.891198457135985
#       z: 0.0
#     orientation: 
#       x: 0.0
#       y: 0.0
#       z: -0.007985089933669315
#       w: 0.9999681186611658
