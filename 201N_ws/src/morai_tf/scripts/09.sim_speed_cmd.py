#! usr/bin/env python3
#-*-coding:utf-8-*-

import rospy
from std_msgs.msg import Float64, Bool

class Sim_cmd_pub:
    def __init__(self):
        rospy.init_node("sim_cmd_node")
        self.pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        rospy.Subscriber('/mission1', Bool, self.sub_CB)
        self.cmd_msg = Float64()
        self.rate = rospy.Rate(10)
        self.speed = 0
        self.flag_param_name = '/navigation_client/flag_param'  # navigation_client.py에서 설정한 파라미터 이름
        self.flag = rospy.get_param(self.flag_param_name)
        print("실행되넹")
        
    def sub_CB(self, msg):
        self.flag = msg.data
        
    def func(self):
        self.speed += 1
        if self.speed >= 2400:
            self.speed = 2400
        self.cmd_msg.data = self.speed
        self.pub.publish(self.cmd_msg)
        print(f"speed:{self.cmd_msg.data}")
        self.rate.sleep()

def main():
    try:
       sim_pub = Sim_cmd_pub()
       while not rospy.is_shutdown() and sim_pub.flag:
           sim_pub.func()
    except rospy.ROSInterruptException:
        pass 
      
if __name__=="__main__":
    main() 
        