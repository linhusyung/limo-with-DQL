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


class environment():
    def __init__(self):
        self.bridge = CvBridge()
        self.reset = reset()
        self.init_word()
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odom)
        self.sub_pose = rospy.Subscriber('/gazebo/model_states', ModelStates, self.get_pose)
        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def init_word(self):
        '''
        重置世界加入终点
        '''
        self.reset.delet_model()
        self.reset.reset_and_stop()
        x = np.random.randint(-1, 1.3)
        y = np.random.randint(-0.3, 2.3)
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
            x = np.random.randint(-2, 1.3)
            y = np.random.randint(-0.3, 2.3)
            self.finish_pose = (x, y)
            current_distance = round(math.hypot(self.finish_pose[0] - self.pose[0],
                                                self.finish_pose[1] - self.pose[1]), 2)
            if current_distance > 0.5:
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

        goal_angle = math.atan2(self.finish_pose[1] - self.position.y, self.finish_pose[0] - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 1)

    def get_pose(self, pose):
        self.pose = (round(pose.pose[2].position.x, 1), round(pose.pose[2].position.y, 1))

    def get_state(self):
        '''
        读取摄影机，lidar咨询，agent位置坐标
        '''

        self.image = None
        while self.image is None:
            try:
                self.image = rospy.wait_for_message('/limo/color/image_raw', Image)
                self.cv_im = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
            except:
                pass

        self.scan = None
        while self.scan is None:
            try:
                self.scan = rospy.wait_for_message('/limo/scan', LaserScan)
                self.scan_ = np.array(self.scan.ranges)
                self.scan_[np.isinf(self.scan_)] = 8
            except:
                pass

        return self.cv_im, self.scan_ / 8, self.pose, self.finish_pose

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
        if current_distance < 0.35:
            self.get_goalbox = True
            done = 1
        return done

    def set_reward(self):
        '''
        设置reward
        '''
        # 距离奖励
        finish_distance = math.dist(self.pose, self.finish_pose)
        distance_reward = -finish_distance*5
        # 角度奖励
        angle_reward = -abs(self.heading)*5

        # 距离障碍物距离
        scan_reward = min(self.scan_)*2

        reward = distance_reward + angle_reward + scan_reward
        # reward = angle_reward + scan_reward

        if self.get_bummper:
            reward = -100
            return reward
        if self.get_goalbox:
            reward = 1000
            return reward
        return reward

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
        next_cv_im, next_scan_, next_pose, next_finish_pose = self.get_state()
        done = self.set_done()
        reward = self.set_reward()
        return next_cv_im, next_scan_, next_pose, next_finish_pose, reward, done

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
