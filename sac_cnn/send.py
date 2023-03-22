#!/usr/bin/env python
# license removed for brevity
import numpy as np
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from Env import environment
# from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
import cv2
import csv
import torch
from net_Lin import *


class agent():
    def __init__(self, num_state, num_action, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor_net(num_state, num_action).to(self.device)
        self.actor.load_state_dict(torch.load(path))

    def get_state(self, scan_, taget) -> torch.tensor:
        # pose_finish_pose = torch.cat((self.data_to_tensor(pose), self.data_to_tensor(finish_pose)), -1)
        return torch.cat((self.data_to_tensor(scan_), self.data_to_tensor(taget)), -1)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def image_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 1, 0))).float().to(self.device)

    def tensor_to_numpy(self, data):
        return data.detach().cpu().numpy()

    def np_to_tensor(self, data):
        return torch.from_numpy(data).float().to(self.device)


def movebase_client(finish_pose):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = finish_pose[0]
    goal.target_pose.pose.position.y = finish_pose[1]
    goal.target_pose.pose.orientation.z = 0
    goal.target_pose.pose.orientation.w = 1

    client.send_goal(goal)

    # client.wait_for_result()


class sub():
    def __init__(self):
        self.x = 0
        self.z = 0

    def get_target_vel(self, data):
        self.x = data.linear.x
        self.z = data.angular.z


def save_variable(i_list, mean_reward, reward_list):
    with open('result/2_3/navigation_with_sac_imitate_test.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入一列資料
        writer.writerow(['玩的次数', i_list])
        writer.writerow(['平均奖励', mean_reward])
        writer.writerow(['奖励加总', reward_list])


if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    movebase_client([-2,0])
    env = environment()
    num_action = 2
    num_state = 25
    path = 'result/2_3/imitate_model.pth'
    # fine_tuning_model_path='result/2_3/model_params_fine_tuning.pth'
    a = agent(num_state, num_action, path)
    rate = rospy.Rate(30)
    sub = sub()
    # b = 0
    # c = 0
    epoch_list = []
    reward_list_ = []
    reward_list_mean = []
    for i in range(10):
        epoch_list.append(i)
        reward_list = []
        episode_step = 0
        while True:
            Target, scan_, pose, finish_pose, state_image = env.get_state()
            movebase_client(env.finish_pose)
            # cmd_vel = rospy.Subscriber("cmd_vel", Twist, sub.get_target_vel)
            # print(Target)
            if Target != -1:
                get_it = True
            else:
                get_it = False

            re_data = []
            for _ in range(24):
                re_data.append(scan_[_ * (len(scan_) // 24)])
            state = torch.cat(
                (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
            action, _ = a.actor(state)
            action = a.tensor_to_numpy(action.squeeze())

            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_state_image = env.text_step(action,
                                                                                                                 get_it)
            print(reward)
            # if Target != -1:
            #     """
            #     target=[[sub.x(action.v),sub.z(action.w)]
            #             [reward,0]]
            #     """
            #     target = np.zeros((2, 2))
            #     # print(sub.x, sub.z)
            #     target[0][0] = sub.x
            #     target[0][1] = sub.z
            #     target[1][0] = reward
            #
            #     # cv2.imwrite('./data/image/image_' + str(b) + '.jpg', state_image)
            #
            #     np.save('./data/image_target_test/image_target_' + str(b) + '.npy', Target)
            #     np.save('./data/scan_test/scan_' + str(b) + '.npy', scan_)
            #     np.save('./data/target_test/target_' + str(b) + '.npy', target)
            #
            #     # cv2.imwrite('./data/next_image/next_state_image' + str(b) + '.jpg', state_image)
            #     # np.save('./data/next_scan/scan_' + str(b) + '.npy', scan_)
            #     # np.save('./data/next_scan/scan_' + str(b) + '.npy', scan_)
            #
            #     cv2.imshow('img', state_image)
            #     cv2.waitKey(1)
            #     print('b',b)
            #     b += 1
            # print(c)
            # c += 1
            # episode_step += 1
            # print(episode_step)
            # if episode_step == 50:
            #     env.get_bummper = True
            #     reward = -50
            reward_list.append(reward)
            if env.get_goalbox:
                env.chage_finish()
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                break

            if env.get_bummper:
                env.init_word()
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                break
            rate.sleep()
        # save_variable(epoch_list, reward_list_mean, reward_list_)
