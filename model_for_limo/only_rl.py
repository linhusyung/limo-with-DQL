#!/usr/bin/env python
# license removed for brevity
import rospy
import actionlib
import cv2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalID
import time
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from net import Actor_net
from std_msgs.msg import String
import math


class c():
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_ = []
        self.cancel_ = 1
        self.bridge = CvBridge()

        self.actor = Actor_net(25, 2).to(self.device)
        self.actor.load_state_dict(torch.load('./model/sac_pose24_1.pth'))
        self.Target_in = 50

    def sub_topic(self):
        self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.get_scan)
        self.sub_done = rospy.Subscriber('/chatter', String, self.get_done)

        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def get_done(self, done):
        self.done = done.data

    def get_scan(self, scan):
        """
        scan咨询
        """
        c=[]
        for i in scan.ranges:
            if i !=0:
                c.append(i)
        self.re_data = []
        for _ in range(24):
            self.re_data.append(c[_ * (len(c) // 24)] / 12)

    def get_image(self, image):
        """
        摄影机咨询
        """
        self.x_f = -1
        self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        # with torch.no_grad():
        #     results = self.yolov7(self.cv_im)
        #     bboxes = np.array(results.pandas().xyxy[0])
        #
        # for i in bboxes:
        #     if i[4] > 0.1:
        #         x_f = (i[2] - i[0]) / 2 + i[0]
        #         y_f = (i[3] - i[1]) / 2 + i[1]
        #         self.bboxes_show(self.cv_im, i, (int(x_f), int(y_f)))
        #         self.x_f = int(x_f * (100 / 640))
        cv2.imshow('img', self.cv_im)
        cv2.waitKey(1)

        action = self.sac()
        print(action)
        self.perform_action(action)
        # if self.done == '1':
        #     print('f')
        #     self.stop()
        #     self.perform_action([0, 0])
        #     self.RL = False
        #     self.stop__ = False

    def get_pose(self, data):
        self.amcl_pose = [data.pose.pose.position.x, data.pose.pose.position.y]

    def goal(self, data):
        self.goal_ = [data.goal.target_pose.pose.position.x, data.goal.target_pose.pose.position.y]

    def cancel(self, data):
        self.cancel_ = data.stamp

    def bboxes_show(self, img, bbox, midpoint):
        cv2.circle(img, midpoint, 1, (0, 0, 255), 4)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 4)
        cv2.putText(img, bbox[6], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(bbox[4]), (int(bbox[2]), int(bbox[3]) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)

    def sac(self):
        # if self.x_f != -1:
        #     self.Target_in = self.x_f
        self.Target_in = 36
        state = torch.cat(
            (self.data_to_tensor(self.Target_in).unsqueeze(0).unsqueeze(0),
             self.data_to_tensor(self.re_data).unsqueeze(0)),
            1)
        print('state', state)
        action, _ = self.actor(state)
        action = self.tensor_to_numpy(action.squeeze())
        return action

    def tensor_to_numpy(self, data):
        return data.detach().cpu().numpy()

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

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def main(self):
        self.sub_topic()


if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    c = c()
    c.main()
    rospy.spin()
