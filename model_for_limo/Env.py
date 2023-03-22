#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import cv2
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
import numpy as np
from math import dist, pi
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
import time
from collections import deque

# np.random.seed(99)


class environment():
    def __init__(self):
        self.bridge = CvBridge()
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odom)
        self.sub_image = rospy.Subscriber('/limo/color/image_raw', Image, self.get_image)
        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.Target = [0, 0, 0]
        # self.state_image = np.zeros((100, 100))
        im = self.bridge.imgmsg_to_cv2(rospy.wait_for_message('/limo/color/image_raw', Image), 'bgr8')
        self.state_image = cv2.resize(im, (100, 100))
        self.action_buffer = deque([], maxlen=2)


    def get_odom(self, odom):
        """
        监听里程计数据
        """
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

    def get_image(self, image):
        """
        监听摄影机数据并转换为目标点在左边还是右边
        """
        self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        image = cv2.resize(self.cv_im, (100, 100))
        self.state_image = image
        mask = self.img_filter(image)

        a = np.where(mask == 255)
        self.c = len(a[0])
        if len(a[0]) != 0:
            x1 = np.argmin(a[1])
            x2 = np.argmax(a[1])
            middle = (a[1][x2] - a[1][x1]) // 2 + a[1][x1]
            self.Target = middle

            cv2.circle(mask, (a[1][x1], a[0][x1]), 1, (255, 0, 255), 4)
            cv2.circle(mask, (a[1][x2], a[0][x2]), 1, (255, 0, 255), 4)
            cv2.circle(mask, (middle, a[0][x1]), 1, (255, 0, 255), 4)

        else:
            self.Target = -1

    def img_filter(self, img):
        '''
        把影像处理成看到背景全部过滤掉只留下红色终点
        '''
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([160, 50, 0])
        upper_red1 = np.array([179, 255, 255])
        lower_red2 = np.array([0, 50, 0])
        upper_red2 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        return mask

    def get_state(self):
        '''
        读取摄影机，lidar咨询，agent位置坐标
        '''

        # self.image = None
        # while self.image is None:
        #     try:
        #         self.image = rospy.wait_for_message('/limo/color/image_raw', Image)
        #         self.cv_im = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        #     except:
        #         pass

        self.scan = None
        while self.scan is None:
            try:
                self.scan = rospy.wait_for_message('/limo/scan', LaserScan)
                self.scan_ = np.array(self.scan.ranges)
                self.scan_[np.isinf(self.scan_)] = 12
            except:
                pass

        # return self.Target, self.scan_ / 8, self.pose, self.finish_pose, self.state_image
        return self.Target, self.scan_ / 12, self.pose, self.finish_pose, self.heading

    def set_done(self):
        '''
        判断是否碰撞和达到终点
        '''
        done = 0
        self.get_bummper = rospy.get_param('/done')
        if self.get_bummper:
            done = 1
        self.get_goalbox = False
        current_distance = round(math.hypot(self.finish_pose[0] - self.pose[0],
                                            self.finish_pose[1] - self.pose[1]), 2)
        # print('current_distance',current_distance)
        # print(self.finish_pose, self.pose)
        if current_distance < 0.3:
            self.get_goalbox = True
            done = 1
        return done

    def set_reward(self):
        '''
        设置reward
        '''
        # 距离奖励
        finish_distance = math.dist(self.pose, self.finish_pose)
        # distance_reward = -finish_distance
        # 角度奖励
        # angle_reward = -1.5 * abs(self.heading)

        # 距离障碍物距离
        # scan_reward = min(self.scan_) * 5
        # scan_reward = 0
        # if min(self.scan_) < 1:
        #     scan_reward = -5
        #     if min(self.scan_) < 0.5:
        #         scan_reward = -10

        # reward = distance_reward + angle_reward + scan_reward
        reward = (1 / finish_distance) - abs(self.heading)

        if self.get_bummper:
            reward = -50
            return reward
        if self.get_goalbox:
            reward = 50
            return reward
        return reward

    def perform_action(self, action):
        '''
        连续动作空间
        v=[0~0.22]
        W=[-2.5~2.5]
        action=(w,v)
        '''
        self.msg.linear.x = float(action[1])
        self.msg.angular.z = float(action[0])
        self.pub.publish(self.msg)

    def chage_reward(self, action):
        self.action_buffer.append(action)
        if self.c != 0:
            try:
                reward = -abs(self.action_buffer[0][0] - self.action_buffer[1][0])
            except:
                reward = 0
        else:
            reward = 0

        if self.get_bummper:
            reward = -50
            return reward
        if self.get_goalbox:
            reward = 50
            return reward
        return reward

    def step(self, action, chage_rew):
        self.perform_action(action)
        next_Target, next_scan_, next_pose, next_finish_pose, heading = self.get_state()
        done = self.set_done()
        if chage_rew:
            reward = self.set_reward()
            # print('没改变')
        else:
            reward = np.float(self.chage_reward(action))

        return next_Target, next_scan_, next_pose, next_finish_pose, reward, done, heading

    def text_step(self, action, Have_it: bool):
        if Have_it:
            self.perform_action(action)
        else:
            pass
        next_Target, next_scan_, next_pose, next_finish_pose, heading = self.get_state()
        done = self.set_done()
        reward = self.set_reward()

        return next_Target, next_scan_, next_pose, next_finish_pose, reward, done, heading

