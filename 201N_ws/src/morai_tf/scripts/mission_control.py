#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
import subprocess

class MissionControl:
    def __init__(self):
        rospy.init_node('mission_control')
        rospy.Subscriber('/mission1', Bool, self.mission_callback)
        self.current_process = None

    def mission_callback(self, msg):
        if msg.data:  # /mission1이 True이면
            if self.current_process is not None:
                rospy.loginfo('Stopping the current process...')
                self.current_process.terminate()

            rospy.loginfo('Starting 09.sim_speed_cmd.py...')
            self.current_process = subprocess.Popen(["rosrun", "morai_tf", "09.sim_speed_cmd.py"])
        else:  # /mission1이 False이면
            if self.current_process is not None:
                rospy.loginfo('Stopping the current process...')
                self.current_process.terminate()
                self.current_process = None

            rospy.loginfo('Starting navi_local_client.py...')
            self.current_process = subprocess.Popen(["rosrun", "morai_tf", "navi_local_client.py"])

if __name__ == "__main__":
    MissionControl()
    rospy.spin()
