#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import cv2
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
import numpy as np
from respawn import *
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
        self.reset = reset()
        self.init_word()
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odom)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.get_pose)
        self.sub_image = rospy.Subscriber('/limo/color/image_raw', Image, self.get_image)
        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.Target = [0, 0, 0]
        # self.state_image = np.zeros((100, 100))
        im = self.bridge.imgmsg_to_cv2(rospy.wait_for_message('/limo/color/image_raw', Image), 'bgr8')
        self.state_image = cv2.resize(im, (100, 100))
        self.action_buffer = deque([], maxlen=2)

    def init_word(self):
        '''
        重置世界加入终点
        '''
        self.reset.delet_model()
        self.reset.reset_and_stop()
        # x = np.random.randint(-1, 1.3)
        # y = np.random.randint(-0.3, 2.3)

        x = 0
        y = 1

        # x_ = [0, 2, 4]
        # # y_ = [1, 0, -1, -2, -3]
        # y_ = [-1]
        # #
        # x = np.random.choice(x_)
        # y = np.random.choice(y_)

        self.reset.SpawnModel(x, y)
        rospy.set_param('/done', 0)
        self.finish_pose = (x, y)

    def chage_finish(self):
        '''
        只改变终点的位置但不改变智能体位置
        '''
        self.reset.stop()
        self.reset.delet_model()
        while True:
            # x = np.random.randint(-2, 1.3)
            # y = np.random.randint(-0.3, 2.3)
            x = 0
            y = 1

            # x_ = [0, 2, 4]
            # y_ = [1, 0, -1, -2, -3]
            #
            # x = np.random.choice(x_)
            # y = np.random.choice(y_)

            self.finish_pose = (x, y)
            current_distance = round(math.hypot(self.finish_pose[0] - self.pose[0],
                                                self.finish_pose[1] - self.pose[1]), 2)
            if current_distance > 0.3:
                break
        self.reset.SpawnModel(x, y)

    def get_odom(self, odom):
        """
        监听里程计数据
        """
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.yaw = yaw
        goal_angle = math.atan2(self.finish_pose[1] - self.position.y, self.finish_pose[0] - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 1)

    def get_pose(self, pose):
        '''
        静态正方地图
        '''
        # self.pose = (round(pose.pose[2].position.x, 1), round(pose.pose[2].position.y, 1))
        '''
        动态地图 
        '''
        self.pose = (round(pose.pose[-2].position.x, 1), round(pose.pose[-2].position.y, 1))
        # print('pose',self.pose)

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

        # cv2.imshow('img', mask)
        # cv2.waitKey(1)

    def find_x1y1(self, mask):
        for _ in range(mask.shape[1]):
            for __ in range(mask.shape[0]):
                if mask[__][_] != 0:
                    return _, __

    def find_x2y2(self, mask):
        for _ in range(mask.shape[1] - 1, -1, -1):
            for __ in range(mask.shape[0] - 1, -1, -1):
                if mask[__][_] != 0:
                    return _, __

    def yolo_bbox_midpoint(self, im_x, midpoint):
        if midpoint <= im_x.shape[1] // 3:
            return [1, 0, 0]
        if midpoint > im_x.shape[1] // 3 and midpoint <= im_x.shape[1] // 3 * 2:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    # def get_image(self, image):
    #     """
    #     监听摄影机数据并转换为目标点在左边还是右边
    #     """
    #     self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
    #     image = cv2.resize(self.cv_im, (100, 100))
    #     self.state_image = image
    #     mask = self.img_filter(image)
    #     a = np.where(mask == 255)
    #     if len(a[0]) != 0:
    #         x1, y1 = self.find_x1y1(mask)
    #         x2, y2 = self.find_x2y2(mask)
    #         middle = (x2 - x1) // 2 + x1
    #         self.Target = middle
    #         # cv2.circle(mask, (x1, y1), 1, (255, 0, 255), 4)
    #         # cv2.circle(mask, (x2, y2), 1, (255, 0, 255), 4)
    #         # cv2.circle(mask, (middle, y2), 1, (255, 0, 255), 4)
    #
    #     else:
    #         self.Target = -1
    #     # cv2.imshow('img', mask)
    #     # cv2.waitKey(1)

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
                self.scan_[np.isinf(self.scan_)] = -1
            except:
                pass

        # return self.Target, self.scan_ / 12, self.pose, self.finish_pose, self.state_image
        finish_distance = sum(np.abs(np.array(self.pose) - np.array(self.finish_pose)))
        return self.Target, self.scan_ / 12, self.pose, self.finish_pose, self.heading, finish_distance

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
        if self.pose[0] > 7:
            self.get_goalbox = True
            done = 1
        return done

    def set_reward(self, action):
        '''
        设置reward
        '''
        # 距离奖励
        finish_distance = math.dist(self.pose, self.finish_pose)
        # finish_distance = sum(np.abs(np.array(self.pose) - np.array(self.finish_pose)))
        # distance_reward = -finish_distance
        # 角度奖励
        # angle_reward = -1.5 * abs(self.heading)

        # 距离障碍物距离
        # scan_reward = min(self.scan_) * 5
        scan_reward = 0
        if min(self.scan_) < 0.2:
            print('1')
            scan_reward = -1
            if min(self.scan_) < 0.15:
                print('2')
                scan_reward = -2

        # reward = distance_reward + angle_reward + scan_reward
        reward = (1 / finish_distance) - abs(self.heading) + scan_reward + action[1]
        # reward = (1 / finish_distance) - abs(self.heading) + scan_reward
        # dis = abs(7 - self.pose[0])
        # reward = scan_reward + action[1] - dis
        # + 10 * 1 / (dis + 1)
        # if action[1] < 1:
        #     reward = scan_reward - 0.1

        if self.get_bummper:
            reward = -50
            # return reward
            # if self.pose[0] > 2:
            #     reward = -300
            # if self.pose[0] > 4:
            #     reward = -200

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
        # self.perform_action(action)
        next_Target, next_scan_, next_pose, next_finish_pose, heading, finish_distance = self.get_state()
        done = self.set_done()
        reward = self.set_reward(action)

        return next_Target, next_scan_, next_pose, next_finish_pose, reward, done, heading, finish_distance

    def text_step(self):

        next_Target, next_scan_, next_pose, next_finish_pose, heading = self.get_state()
        done = self.set_done()
        reward = self.set_reward()

        return next_Target, next_scan_, next_pose, next_finish_pose, reward, done, heading

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
