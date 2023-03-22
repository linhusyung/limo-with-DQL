import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pynput import keyboard
import torch
import time
import cv2
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError

from respawn import *

import time


class img():
    def __init__(self):
        self.bridge = CvBridge()
        # self.train_index = 0
        # self.text_index = 0
        # self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', './my_data_tiny/best.pt', source='local',
        #                              force_reload=False)
        # self.yolov7.eval()
        # self.last_time = time.time()
        # self.fps_count = 0
        #
        # self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        self.sub_image = rospy.Subscriber('/limo/color/image_raw', Image, self.get_image)


    def get_image(self, image):
        cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        cv2.imshow('img', cv_im)
        cv2.waitKey(1)

        mask = self.img_filter(cv_im)

        a = np.where(mask == 255)
        self.c = len(a[0])
        if len(a[0]) != 0:
            x1 = np.argmin(a[1])
            x2 = np.argmax(a[1])
            middle = (a[1][x2] - a[1][x1]) // 2 + a[1][x1]
            self.Target = middle

            cv2.circle(mask, (a[1][x1], a[0][x1]), 1, (255, 0, 255), 4)
            cv2.circle(mask, (a[1][x2], a[0][x2]), 1, (255, 0, 255), 4)
            cv2.circle(mask, (middle, a[0][x1]), 1, (100, 100, 100), 10)

        else:
            self.Target = -1

        cv2.imshow('mask', mask)
        cv2.waitKey(1)
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

if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    i = img()
    rospy.spin()
