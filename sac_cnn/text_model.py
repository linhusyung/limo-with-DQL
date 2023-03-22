#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
from Env import environment
# from net_Lin import *
from imitate_networl import *
import matplotlib.pyplot as plt
import csv
import cv2
import random
from network_1 import *


class agent():
    def __init__(self, num_state, num_action, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor_net(num_state, num_action).to(self.device)
        self.actor.load_state_dict(torch.load(path))

        # self.look_actor = Actor_net_1(25, num_action).to(self.device)
        # self.look_actor.load_state_dict(torch.load('./result/2_3/imitate_model.pth'))

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


def save_variable(i_list, mean_reward, reward_list):
    with open('result/2_3/pre.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入一列資料
        writer.writerow(['玩的次数', i_list])
        writer.writerow(['平均奖励', mean_reward])
        writer.writerow(['奖励加总', reward_list])


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(30)
    scan_num = 24
    num_action = 2
    num_state = 1 + scan_num
    # num_state = 1 + scan_num
    # path = 'result/2_3/imitate_model.pth'
    pre_model_path = './result/map4_/sac_1.pth'
    a = agent(num_state, num_action, pre_model_path)
    env = environment()
    epoch_list = []
    reward_list_ = []
    reward_list_mean = []

    for i in range(10):
        action_index = 0
        reward_list = []
        episode_step = 0
        print('第', i, '次')
        while True:
            print('第', action_index, '个动作')
            action_index += 1
            Target, scan_, pose, finish_pose, heading, finish_distance = env.get_state()
            # if Target == -1:
            # finish_pose = np.random.normal(finish_pose, 0.001)
            # pose = np.random.normal(pose, 0.001)
            # finish_rebot_pose = np.hstack((finish_pose, pose))
            # finish_rebot_pose = a.np_to_tensor(finish_rebot_pose).unsqueeze(0)
            heading_ = a.data_to_tensor(heading).unsqueeze(0).unsqueeze(0)

            re_data = []
            for _ in range(scan_num):
                re_data.append(scan_[_ * (len(scan_) // scan_num)])

            state = torch.cat(
                (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
            print(state)
            # state = torch.cat(
            #     (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
            # print(state)
            action, _ = a.actor(state)
            action = a.tensor_to_numpy(action.squeeze())
            # else:
            #     re_data = []
            #     for _ in range(24):
            #         re_data.append(scan_[_ * (len(scan_) // 24)])
            #     state = torch.cat(
            #         (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
            #     action, _ = a.look_actor(state)
            #     action = a.tensor_to_numpy(action.squeeze())

            print(action)

            # print('action', action)
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_heading, next_finish_distance = env.step(
                action,
                True)
            print('reward', reward)
            # episode_step += 1
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
