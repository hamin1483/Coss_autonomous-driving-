#!/usr/bin/env python3

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
import actionlib
from geometry_msgs.msg import PoseWithCovarianceStamped

class NavigationClient():
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.goal_list = []  # 목표 지점 리스트

        # 로봇의 시작 위치를 amcl_pose로 지정
        start_pose_msg = PoseWithCovarianceStamped()
        start_pose_msg.header.frame_id = 'map'
        start_pose_msg.pose.pose.position.x = 0.060722974402695015
        start_pose_msg.pose.pose.position.y = 0.0010099079608249146
        start_pose_msg.pose.pose.orientation.z = 0.03350902978386785
        start_pose_msg.pose.pose.orientation.w = 1.0

        self.set_initial_pose(start_pose_msg)

        # 다음 목표 위치 설정
        self.waypoint_1 = MoveBaseGoal()
        self.waypoint_1.target_pose.header.frame_id = 'map'
        self.waypoint_1.target_pose.pose.position.x = 17.380455503314376
        self.waypoint_1.target_pose.pose.position.y = -8.995952975742531
        self.waypoint_1.target_pose.pose.orientation.z = 0.032762097479707895
        self.waypoint_1.target_pose.pose.orientation.w = 1.0

        self.goal_list.append(self.waypoint_1)

        self.sequence = 0
        self.start_time = rospy.Time.now()

    def set_initial_pose(self, pose_msg):
        rospy.wait_for_service('/initialpose')
        try:
            set_initial_pose = rospy.ServiceProxy('/initialpose', PoseWithCovarianceStamped)
            set_initial_pose(pose_msg)
        except rospy.ServiceException:
            pass

    def run(self):
        if self.client.get_state() != GoalStatus.ACTIVE:  # 돌아가고 있지 않으면 다음 sequence 진행
            self.start_time = rospy.Time.now()
            self.client.send_goal(self.goal_list[self.sequence])
            # self.sequence = (self.sequence+1) % 2 # 0이나 1, waypoint 개수에 맞출것
        else:
            if rospy.Time.now().to_sec() - self.start_time.to_sec() > 30.0:
                self.stop()

    def stop(self):
        self.client.cancel_all_goals()

def main():
    rospy.init_node('navigation_client')
    nc = NavigationClient()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        nc.run()
        rate.sleep()

if __name__ == "__main__":
    main()
