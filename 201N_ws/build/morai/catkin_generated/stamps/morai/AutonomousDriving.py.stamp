import rospy
import csv
from geometry_msgs.msg import PoseWithCovarianceStamped
from morai_msgs.msg import GetTrafficLightStatus
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from math import *
import numpy as np
import time
import cv2
import os

WB = 0.25
Ldc = 0.4 # 0.4
v_gain = 0.3
k_gain = 0.8

class PurePursuitController:
    def __init__(self):
        # 노드 초기화 및 필요한 토픽 및 변수 설정
        rospy.init_node('pure_pursuit_controller')
        self.current_dir = os.getcwd()
        self.path_file = self.current_dir + '/bagfile/path_spline_8.csv'
        print(self.path_file)
        self.waypoints = []
        self.waypoint_x = []
        self.waypoint_y = []
        self.load_waypoints()
        self.k = 0
        self.target_speed = 2400  # 로봇의 목표 속도 (임의의 값, 필요에 따라 조절)
        self.v = 2400
        self.slow_v = 1400
        self.x = 0
        self.y = 0
        self.speed = 0
        self.steer = 0.5
        self.Ld = 0
        self.gx = 0
        self.gy = 0
        self.yaw = 0
        self.delta = 0
        self.er = 0
        self.ind = 0
        
        self.fault_path_flag = False
        
        # 퍼블리셔 설정
        self.speed_pub = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)

        # 서브스크라이버 설정
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.current_pose = PoseWithCovarianceStamped()

        #-------------------Lidar 추가사항---------------------------------------------#
        rospy.Subscriber("/scan", LaserScan, self.lidar_CB)
        self.scan_msg = LaserScan()
        self.obstacle_flag = False
        self.static_obstacle = False
        self.right_line = True
        self.wait_flag = False
        self.obstacle_count = 0    
        self.speed_msg = Float64()
        self.steer_msg = Float64()
        # print(self.current_pose)
        #------------------Cam 추가사항 ----------------------------------#
        rospy.Subscriber("/GetTrafficLightStatus",GetTrafficLightStatus,self.traffic_CB)
        rospy.Subscriber("image_jpeg/compressed",CompressedImage,self.cam_CB)
        self.bridge = CvBridge()
        self.image_msg = CompressedImage()
        self.traffic_msg = GetTrafficLightStatus()
        self.traffic_flag = 0
        self.prev_cross_signal = 0
        self.cross_signal = 0
        self.cross_flag = 0
        self.img = []
        self.img_yellow = []
        self.x = 0
        self.y = 0
        self.center_index = 0
        self.standard_line = 0
        self.degree_per_pixel = 0
        self.cross_cnt = 0 
        self.lane_signal = 0
        self.prev_lane_signal = 0
        self.only_right_lane_cnt = 0
        self.lane_cnt = 0
        self.mission_1 = False
        self.mission_rotary = False
        self.mission_3 = False
        self.mission_go = False
        self.prev_cross_flag = False
        self.mission_cross = False
        self.mrflag = False
        self.round_lidar_flag = False
        
        #### 로터리 플래그 ####
        self.rotary_drive = False
        #### 교차로 플래그 ####
        self.cross_drive = False
        
        self.Mission_round_flag = False
        self.Lidar_ON = True
        self.return_flag = True
        ######## SLAM & Navigation 연결 ########
        rospy.Subscriber("/mission1", Bool, self.mission_CB)
        self.mission_flag = False
        
    def mission_CB(self,msg):
        self.mission_flag = msg.data
        
    ################### Drive 함수 추가 ##########
    
    ###### 로터리 판별 ######
    def is_rotary(self):
        if np.hypot(self.x - self.waypoint_x[1300], self.y - self.waypoint_y[1300]) < 0.2:
            self.rotary_drive = True
            
    ###### 교차로 판별 ######
    def is_cross(self):
        if np.hypot(self.x - self.waypoint_x[1466], self.y - self.waypoint_y[1466]) < 0.15:
            self.cross_drive = True
            self.rotary_drive = False
            self.Lidar_ON = False
            self.slow_v = 1600
    
    def traffic_CB(self,msg): # 신호등 CB 
        #print(msg)
        self.traffic_msg = msg
        if self.traffic_msg.trafficLightIndex == "SN000005":
            self.cross_signal = self.traffic_msg.trafficLightStatus
            if self.prev_cross_signal != self.cross_signal:
                self.prev_cross_signal = self.cross_signal

    def cam_CB(self,msg):
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)
        
    ################### Lidar 함수 추가 ##########
    def lidar_CB(self, msg): # Lidar 콜백 함수 
        self.scan_msg = msg
        self.round_lidar_flag = False
        self.return_flag = True
        if self.Lidar_ON == True:
            self.obstacle()
        else : 
            self.obstacle_flag = False
            self.wait_flag = False
            self.static_obstacle = False

    def obstacle(self) : #물체가 있는지 판단해서 플래그
        degree_min = self.scan_msg.angle_min * 180/pi
        degree_max = self.scan_msg.angle_max * 180/pi
        degree_angle_increment = self.scan_msg.angle_increment * 180/pi
        degrees= [degree_min + degree_angle_increment * index for index, value in enumerate(self.scan_msg.ranges)] #각도값
        degree_array = np.array(degrees)
        obstacle_degrees = []
        obstacle_index = []
        right_line_det = []
        
        for index, value in enumerate(self.scan_msg.ranges):
            #-----------각도, 거리 수정 필요------------------------------------------------------------------------------------------
            if abs(degrees[index])<30 and 0<value<1.4:    # 물체 확인(각도 값 & 거리 값 함께 고려)
                obstacle_degrees.append(degrees[index])
                obstacle_index.append(index)
                if abs(degrees[index]) < 5 and 0.3<value<0.8 : #정면
                    if self.static_obstacle == False:
                        self.wait_flag = True
                        self.obstacle_count += 1
                
            elif -95<degrees[index]<-85 and 1.5<value<2.5 : #차선 확인용 벽면 탐지
                right_line_det.append(degrees[index])
            else:
                pass
            
            if abs(degrees[index])<90 and 0<value<1.0: # 로터리 범위
                self.round_lidar_flag = True
                
            if 60<abs(degrees[index])<90 and 0<value<0.7: # right right
                self.return_flag = False

        try:
            if len(right_line_det) != 0:
                self.right_line = True
            else :
                self.right_line = False
                
            if len(obstacle_degrees) != 0 or self.return_flag == False: ######
                self.obstacle_flag = True
                if self.wait_flag == True :
                #정적, 동적 판단
                    if self.obstacle_count > 150 :
                        self.static_obstacle = True
                        self.wait_flag = False
                    else :
                        pass

            else :
                self.obstacle_flag = False
                self.static_obstacle = False
                self.wait_flag = False
                self.obstacle_count = 0
               
                
        except:
            self.obstacle_flag = False
            self.static_obstacle = False
            self.wait_flag = False
            self.obstacle_count = 0
      
    
    
    def avoid_in_right(self): # 2차선 -> 1차선 
        yaw = self.yaw * 0.8
        t1 = rospy.get_time()
        t2 = rospy.get_time()
        while t2- t1 <= 0.7:
            self.pub_speed_and_steer(1000,0.1 + yaw)
            t2 = rospy.get_time()
           
        t1 = rospy.get_time()
        t2 = rospy.get_time()
        while t2 - t1 <= 1.1: 
            self.pub_speed_and_steer(1000,0.9 - yaw)
            t2 = rospy.get_time()
        
        # t1 = rospy.get_time()
        # t2 = rospy.get_time()
        # while t2- t1 <= 0.4 :
        #     self.pub_speed_and_steer(1000,0.9 + abs(yaw))
        #     t2 = rospy.get_time()
        ######################################
        t1 = rospy.get_time()
        t2 = rospy.get_time()

        while t2 - t1 <= 0.7:
            target_angle = self.calculate_target_angle()
            steer = self.normalize_steer(target_angle)
            self.pub_speed_and_steer(1000,steer)
            t2 = rospy.get_time()
        #######################################
    def avoid_in_left(self): #1차선 -> 2차선 
        t1 = rospy.get_time()
        t2 = rospy.get_time()
        while t2- t1 <= 1.0  :

            self.pub_speed_and_steer(600,0.1)
            t2 = rospy.get_time()
           
        t1 = rospy.get_time()
        t2 = rospy.get_time()

        while t2 - t1 <= 1.0:
            self.pub_speed_and_steer(600,0.9)
            t2 = rospy.get_time()
            
    def pub_speed_and_steer(self,speed,steer): #speed & steer pub
        self.steer_msg.data = steer
        self.speed_msg.data = speed
        self.speed_pub.publish(self.speed_msg)
        self.steer_pub.publish(self.steer_msg)    
        
    def heading_fit(self):
        if self.yaw - self.k > 0.05:
            self.pub_speed_and_steer(300, 0.1)
            rospy.sleep(0.4)
        elif self.yaw - self.k < -0.05:
            self.pub_speed_and_steer(300, 0.9)
            rospy.sleep(0.4)
        else:
            pass
            
    ###########################################
    def load_waypoints(self):
        with open(self.path_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.waypoints.append(row) # 한 줄씩 저장
                self.waypoint_x.append(float(row[0]))
                self.waypoint_y.append(float(row[1]))

    def pose_callback(self, msg):
 
        self.dist = [] # distance 값 list
        self.ind_list = [] # 추종할 범위 내에 들어온 인덱스들 list
        
        self.front_x = [] # 내 앞의 몇 포인트의 점 (x좌표) (곡률 계산용)
        self.back_x = [] # 내 뒤의 몇 포인트의 점 (x좌표) (곡률 계산용)
        
        self.front_y = [] # 내 앞의 몇 포인트의 점 (y좌표) (곡률 계산용)
        self.back_y = [] # 내 뒤의 몇 포인트의 점 (y좌표) (곡률 계산용)
        
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.current_pose = msg
        
        for i in range(0, len(self.waypoint_x)):
            dx = self.x - self.waypoint_x[i]
            dy = self.y - self.waypoint_y[i]
            self.dist.append(np.hypot(dx, dy))
            if np.hypot(self.x - self.waypoint_x[700], self.y - self.waypoint_y[700]) < 1:
                self.fault_path_flag = True
            if np.hypot(self.x - self.waypoint_x[0], self.y - self.waypoint_y[0]) < 0.1:
                self.fault_path_flag = False
            
            if self.Ld*0.8 < self.dist[i] < self.Ld*1.2:
                self.ind_list.append(i)
            else:
                pass
                
        self.er = np.min(self.dist)
        if self.ind_list == []:
            self.ind = np.argmin(self.dist)
            self.gx = self.waypoint_x[self.ind]
            self.gy = self.waypoint_y[self.ind]
        else:
            if not self.fault_path_flag:
                if len(self.ind_list) != 0:
                    filtered = [i for i in self.ind_list if i < 700]
                    self.ind = np.max(filtered)
                else:
                    pass
            else:
                self.ind = np.max(self.ind_list)
            self.gx = self.waypoint_x[self.ind]
            self.gy = self.waypoint_y[self.ind]
        # 곡률 계산 부분
        if self.ind + 20 < len(self.waypoint_x):
            self.front_x = self.waypoint_x[self.ind + 1: self.ind + 20]
            self.front_y = self.waypoint_y[self.ind + 1: self.ind + 20]
        else:
            self.front_x = self.waypoint_x[self.ind + 1: len(self.waypoint_x)]
            self.front_y = self.waypoint_y[self.ind + 1: len(self.waypoint_y)]
        
        if self.ind - 15 > 0:
            self.back_x = self.waypoint_x[self.ind - 15: self.ind]
            self.back_y = self.waypoint_y[self.ind - 15: self.ind]
        else:
            self.back_x = self.waypoint_x[0:self.ind]
            self.back_y = self.waypoint_y[0:self.ind]
        
        front_k, back_k = self.calculate_curvature()
        avg_k = (front_k + back_k) /2.0
        
        if avg_k > 0.3:
            self.k = avg_k
        else:
            self.k = 0
            
        # print(self.k)
        if abs(abs(self.yaw) - abs(self.k)) < 0.12:
            self.target_speed = self.v
        else: 
            self.target_speed = self.slow_v
            # self.target_speed = self.v - (self.k * k_gain * (2500/3.6))
        
        self.Ld = Ldc + ((self.target_speed / 280) / 3.6) * v_gain
        # print(self.target_speed)
        
    def calculate_curvature(self):
        # 점들을 이용하여 곡률 계산
        grad_front = []
        grad_back = []
        front_k = 0
        back_k = 0
        
        for i in range(1, len(self.front_x)):
            dx = self.front_x[i] - self.front_x[i-1]
            dy = self.front_y[i] - self.front_y[i-1]
            
            grad_front.append(abs(atan2(dy,dx)))
            
            front_k = sum(grad_front) / len(grad_front)
            
        for i in range(1, len(self.back_x)):
            dx = self.back_x[i] - self.back_x[i-1]
            dy = self.back_y[i] - self.back_y[i-1]
            
            grad_back.append(abs(atan2(dy,dx)))
            
            back_k = sum(grad_back) / len(grad_back)
        
        return front_k, back_k

    def calculate_target_angle(self):
        q_x = self.current_pose.pose.pose.orientation.x
        q_y = self.current_pose.pose.pose.orientation.y
        q_z = self.current_pose.pose.pose.orientation.z
        q_w = self.current_pose.pose.pose.orientation.w
        
        # 방향 벡터 계산
        direction_vector_1 = (
            2 * (q_w * q_z + q_x * q_y),
            2 * (q_y * q_z - q_w * q_x),
            1 - 2 * (q_y**2 + q_z**2)
        )
        
        self.yaw = atan2(direction_vector_1[0], direction_vector_1[2])
        
        dx = self.gx - self.current_pose.pose.pose.position.x
        dy = self.gy - self.current_pose.pose.pose.position.y
        
        alpha = atan2(dy, dx) - self.yaw
        self.delta = atan2(2.0 * WB * sin(alpha), self.Ld)
        target_angle = self.delta 

        return target_angle

    def normalize_steer(self, angle):
        normalized_steer = -(angle / 0.65) + 0.5
        return normalized_steer

    def control_loop(self):
        self.mission_flag2 = False
        rate = rospy.Rate(20)  # 20Hz로 제어 루프 설정

        while not rospy.is_shutdown():
            if self.mission_flag:
                if not self.mission_flag2:
                    rospy.sleep(1.5)
                    self.pub_speed_and_steer(300, 0.5)
                    rospy.sleep(2.0)
                    self.mission_flag2 = True
                self.is_rotary()
                self.is_cross()

                ######## Obstacle Avodiance ########
                if (self.obstacle_flag == True) and (self.wait_flag == False) and self.static_obstacle == False :
                     # Pure Pursuit 알고리즘을 사용하여 목표 각도 계산
                    target_angle = self.calculate_target_angle()
                    steer = self.normalize_steer(target_angle)
                    self.pub_speed_and_steer(400,steer)    #장애물 있으면 일단 감속
                elif (self.obstacle_flag == True) and (self.wait_flag == True) and (self.static_obstacle == False) : 
                    self.pub_speed_and_steer(0,0.5) #정면에 있으면 정지
                elif (self.static_obstacle == True) : 
                    self.static_obstacle == False
                    self.avoid_in_right()
                    target_angle = self.calculate_target_angle()
                    steer = self.normalize_steer(target_angle)
                    self.pub_speed_and_steer(400,steer)
                    
                ######## Pure Puresuit ########
                else :
                    if self.current_pose is not None:
                        # 현재 위치에서 가장 가까운 목표점 선택

                        # Pure Pursuit 알고리즘을 사용하여 목표 각도 계산
                        target_angle = self.calculate_target_angle()
                        steer = self.normalize_steer(target_angle)

                        if self.cross_drive and self.rotary_drive == False:
                            if self.cross_signal != 33:
                                self.speed = 0
                            else:
                                self.speed = 1000
                                self.cross_drive = False
                            speed = self.speed
                            self.speed_pub.publish(speed)
                            self.steer_pub.publish(steer)

                        elif self.rotary_drive and self.cross_drive == False: # 로터리 구간 
                            if self.round_lidar_flag:
                                self.speed = 0
                            else:
                                self.speed = 1000
                                # print('로터리')
                            speed = self.speed
                            self.speed_pub.publish(speed)
                            self.steer_pub.publish(steer)
                        else:
                            speed = self.target_speed
                            self.speed_pub.publish(speed)
                            self.steer_pub.publish(steer)

                        # 종점에서 멈추는 부분
                        if np.hypot(self.x - self.waypoint_x[-1], self.y - self.waypoint_y[-1]) < 0.5:
                            self.speed_pub.publish(0) 
                        # 일반 주행
                        else:
                            self.speed_pub.publish(speed)
                            self.steer_pub.publish(steer)
            else:
                pass
            rate.sleep()

def main():
    controller = PurePursuitController()
    print(controller.mission_flag)
    controller.control_loop()

if __name__ == '__main__':
    main()