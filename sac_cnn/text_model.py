#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
from Env import environment
from net_Lin import *
import matplotlib.pyplot as plt
import csv
import cv2
import random


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


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
    num_action = 2
    num_state = 24 + 3
    path='model/model_params.pth'
    # path='model/model_best_.pth'
    a = agent(num_state, num_action,path)
    env = environment()

    for i in range(10):
        action_index = 0
        while True:
            print('第', action_index, '个动作')
            action_index += 1
            Target, scan_, pose, finish_pose, state_image = env.get_state()

            state = torch.cat((a.data_to_tensor(Target).unsqueeze(0), a.data_to_tensor(scan_).unsqueeze(0)), 1)
            action, _ = a.actor(state)

            # print(Target)
            action = a.tensor_to_numpy(action.squeeze())
            print('action', action)
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_state_image = env.step(action)
            print('reward', reward)
            rate.sleep()

            if env.get_goalbox:
                env.chage_finish()
                break

            if env.get_bummper:
                env.init_word()
                break
