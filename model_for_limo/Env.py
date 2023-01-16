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
from nav_msgs.msg import Odometry


class environment():
    def __init__(self):
        self.bridge = CvBridge()

        self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.Target = [0, 0, 0]
        # self.state_image = np.zeros((100, 100))
        im = self.bridge.imgmsg_to_cv2(rospy.wait_for_message('/camera/rgb/image_raw', Image), 'bgr8')
        self.state_image = im

    def get_image(self, image):
        """
        监听摄影机数据并转换为目标点在左边还是右边
        """
        self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        image = cv2.resize(self.cv_im, (100, 100))
        self.state_image = self.cv_im
        mask = self.img_filter(image)
        a = np.where(mask == 255)

        left = len(np.where(a[1] < 50)[0])
        rigth = len(np.where(a[1] > 50)[0])

        self.Target = [0, 0, 0]

        if left > rigth:
            self.Target = [1, 0, 0]
        elif left < rigth:
            self.Target = [0, 0, 1]
        elif left != 0:
            if rigth != 0:
                self.Target = [0, 1, 0]

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

        # cv2.imshow('img', mask)
        # cv2.waitKey(1)
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
                self.scan = rospy.wait_for_message('/scan', LaserScan)
                self.scan_ = np.array(self.scan.ranges)
                self.scan_[np.isinf(self.scan_)] = 12
            except:
                pass

        # return self.Target, self.scan_ / 8, self.pose, self.finish_pose, self.state_image
        return self.Target, self.scan_ / 12, self.state_image

    def perform_action(self, action):
        '''
        连续动作空间
        v=[0~0.5]
        W=[-2.5~0.22]
        action=(w,v)
        '''
        self.msg.linear.x = float(action[1])
        self.msg.angular.z = float(action[0])
        self.pub.publish(self.msg)

    def step(self, action):
        self.perform_action(action)
        next_Target, next_scan_, state_image = self.get_state()

        return next_Target, next_scan_, state_image

# if __name__ == '__main__':
#     rospy.init_node('text_listener', anonymous=True)
#     rate = rospy.Rate(1)
#     env = environment()
#     for i in range(50):
#         reward_list = []
#         while True:
#             cv_im, scan_, pose, finish_pose = env.get_state()
#             out = np.random.randint(0, 5)
#             print('min_scan',min(scan_))
#             next_cv_im, next_scan_, next_pose, next_finish_pose, reward, done=env.step(out)
#             print('done',done)
#             print('reward',reward)
#             reward_list.append(reward)
#             if env.get_goalbox:
#                 env.chage_finish()
#                 print('reward_list',sum(reward_list))
#                 break
#             if env.get_bummper:
#                 env.init_word()
#                 print('reward_list',sum(reward_list))
#                 break
#
#             rate.sleep()
