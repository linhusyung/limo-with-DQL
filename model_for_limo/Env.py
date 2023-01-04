#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import cv2
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
import numpy as np
from nav_msgs.msg import Odometry


class environment():
    def __init__(self):
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        # self.sub_image = rospy.Subscriber('/limo/color/image_raw', Image, self.get_image)
        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        im = self.bridge.imgmsg_to_cv2(rospy.wait_for_message('/camera/rgb/image_raw', Image), 'bgr8')
        self.state_image = cv2.resize(im, (100, 100))

    def get_image(self, image):
        """
        监听摄影机数据并转换为目标点在左边还是右边
        """
        self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        image = cv2.resize(self.cv_im, (100, 100))
        self.state_image = image
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
        # cv2.imshow('mask', mask)
        # cv2.waitKey(1)
        cv2.imshow('img', img)
        cv2.waitKey(1)
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
                # self.scan = rospy.wait_for_message('/limo/scan', LaserScan)
                self.scan_ = np.array(self.scan.ranges)
                self.scan_[np.isinf(self.scan_)] = 8
            except:
                print('scan_topic')

        return self.Target, self.scan_ / 8, self.state_image

    def get_action(self, Q_index):
        '''
        离散动作空间
        1:v=0.3,w=1
        2:v=0.3,w=-1
        3:v=0.15,w=2
        4:v=0.15,w=-2
        5:v=0.3,w=0
        '''
        self.action_space = ((0.15, 0.75), (0.15, -0.75), (0.15, 1.5), (0.15, -1.5), (0.15, 0))
        self.perform_action(self.action_space[Q_index])

    def perform_action(self, action):
        """
        执行动作
        """
        self.msg.linear.x = float(action[0])
        self.msg.angular.z = float(action[1])
        self.pub.publish(self.msg)

    def step(self, out: int):
        self.get_action(out)
        next_Target, next_scan_, state_image = self.get_state()

        return next_Target, next_scan_
