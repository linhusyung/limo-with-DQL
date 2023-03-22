#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
import random
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class get_d():
    def __init__(self):
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber('/camera/color/image_raw', Image, self.get_image)
        self.pub = rospy.Publisher('/chatter', String, queue_size=10)


    def get_image(self, im):
        # self.pub.publish('0')
        cv_im = self.bridge.imgmsg_to_cv2(im, 'bgr8')
        cv2.imshow('img', cv_im)
        cv2.waitKey(1)
        mask = self.img_filter(cv_im)

        a = np.where(mask == 255)
        print(len(a[0]))
        if len(a[0]) <= 20000:
            print('done')
            self.pub.publish('1')
        else:
            self.pub.publish('0')

        cv2.imshow('mask', mask)
        cv2.waitKey(1)
        # print(cv_im)

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


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    get_d = get_d()
    rospy.spin()
