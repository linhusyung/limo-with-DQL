# 測試
import cv2
import rospy
from tf2_msgs.msg import TFMessage
from gazebo_msgs.msg import ModelStates
import math
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
import numpy as np


class text():
    def __init__(self):
        rospy.init_node('text_listener', anonymous=True)
        self.bridge = CvBridge()

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
        return mask

    def sub(self):
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.callback)

    def callback(self, data):
        # lower_red2 = np.array([0, 50, 0])
        # upper_red2 = np.array([10, 255, 255])
        cv_im = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        image = cv2.resize(cv_im, (100, 100))

        mask = self.img_filter(image)
        a = np.where(mask == 255)

        left = len(np.where(a[1] < 50)[0])
        rigth = len(np.where(a[1] > 50)[0])
        if left == 0:
            if rigth == 0:
                print([0, 0, 0])

        if left > rigth:
            print([1, 0, 0])
        elif left < rigth:
            print([0, 0, 1])
        elif left == rigth:
            print([0, 1, 0])

        cv2.imshow('mask', mask)
        cv2.waitKey(1)


if __name__ == '__main__':
    t = text()
    t.sub()
    rospy.spin()
