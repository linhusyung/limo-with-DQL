#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
from Env import environment
from network import *
import matplotlib.pyplot as plt
import csv
import cv2


class agent():
    def __init__(self):
        self.model = DDQN(state_dim=24 + 4)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PATH = 'model/model_params.pth'

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float)

    def get_state(self, scan_, pose, finish_pose) -> torch.tensor:
        pose_finish_pose = torch.cat((self.data_to_tensor(pose), self.data_to_tensor(finish_pose)), -1)
        return torch.cat((self.data_to_tensor(scan_), pose_finish_pose), -1)

    def choose_action(self, out: torch.tensor) -> int:
        return int(out.argmax().cpu().numpy())

if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(1)
    a = agent()
    env = environment()
    b_list = []
    reward_list_ = []
    reward_list_mean = []
    count = 0
    for i in range(1000):
        reward_list = []
        b_list.append(i)
        print('第', i, '次游戏')
        episode_step = 0
        action_index = 0
        while True:
            print('第', action_index, '个动作')
            print('count', count)
            action_index += 1
            cv_im, scan_, pose, finish_pose = env.get_state()
            state = a.get_state(scan_, pose, finish_pose)

            a.model.load_state_dict(torch.load(a.PATH))
            out = a.model(state).to(a.device)
            action = a.choose_action(out)
            print(out, action)

            next_cv_im, next_scan_, next_pose, next_finish_pose, reward, done = env.step(action)

            print(reward)
            reward_list.append(reward)

            episode_step += 1
            if episode_step == 300:
                env.get_goalbox = True

            if env.get_goalbox:
                env.chage_finish()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                break
            if env.get_bummper:
                env.init_word()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                break

            rate.sleep()