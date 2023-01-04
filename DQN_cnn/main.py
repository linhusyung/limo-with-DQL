#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
import torch

from Env import environment
from network import *
import matplotlib.pyplot as plt
import csv
import cv2
import time


class agent():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DDQN_Global().to(self.device)
        self.target_model = DDQN_Global().to(self.device)

        self.eps = 1
        self.batch_size = 64
        self.buffer = Replay_Buffers(batch_size=self.batch_size)
        self.gamma = 0.99
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def image_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 1, 0))).float().to(self.device)

    def get_state(self, scan_, taget) -> torch.tensor:
        # pose_finish_pose = torch.cat((self.data_to_tensor(pose), self.data_to_tensor(finish_pose)), -1)

        return torch.cat((self.data_to_tensor(scan_), self.data_to_tensor(taget)), -1)

    def choose_action(self, out: torch.tensor) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(0, 5)
        else:
            return int(out.argmax().cpu().numpy())

    def tuple_of_tensors_to_tensor(self, tuple_of_tensors):
        return torch.stack(tuple_of_tensors, dim=0).squeeze()

    def replay_resize(self, replay):
        state_img = torch.zeros(self.batch_size, 3, 100, 100).to(self.device)
        state_scan = torch.zeros(self.batch_size, 24).to(self.device)
        state = (state_img, state_scan)

        state_img_next = torch.zeros(self.batch_size, 3, 100, 100).to(self.device)
        state_scan_next = torch.zeros(self.batch_size, 24).to(self.device)
        next_state = (state_img_next, state_scan_next)

        action = torch.zeros(self.batch_size, 1, dtype=torch.int64).to(self.device)
        reward = torch.zeros(self.batch_size, 1).to(self.device)
        done = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(self.device)
        for _ in range(len(replay)):
            state[0][_] = replay[_]['state'][0]
            state[1][_] = replay[_]['state'][1]

            next_state[0][_] = replay[_]['next_state'][0]
            next_state[1][_] = replay[_]['next_state'][1]

            action[_] = replay[_]['action']

            reward[_] = replay[_]['reward']

            done[_] = replay[_]['done']
        return state, next_state, action, reward, done

    def train(self, replay):
        print('train')
        state, next_state, action, reward, done = self.replay_resize(replay)

        Q = self.model(state)
        Q_next = self.model(next_state)
        argmaz_action = torch.argmax(Q_next, 1).unsqueeze(1)

        Q_target_net = self.target_model(next_state)
        max_Q_ = Q_target_net.gather(1, argmaz_action)

        Q_target = reward + self.gamma * max_Q_ * (1 - done)
        Q_eval = Q.gather(1, action)
        #
        loss = self.loss_fn(Q_target, Q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_variable(self, i_list, mean_reward, reward_list):
        with open('result/cnn.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            # 寫入一列資料
            writer.writerow(['玩的次数', i_list])
            writer.writerow(['平均奖励', mean_reward])
            writer.writerow(['奖励加总', reward_list])

    def save_(self):
        torch.save(self.model.state_dict(), 'model/model_Global.pth')
        torch.save(self.model.cnn.state_dict(), 'model/model_cnn.pth')
        torch.save(self.model.Linear.state_dict(), 'model/model_Linear.pth')


    def save_best(self):
        print('储存最好的model')
        torch.save(self.model.state_dict(), 'model/model_best_cnn.pth')


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
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
        print(a.eps)
        episode_step = 0
        action_index = 0
        while True:
            print('第', action_index, '个动作')
            print('count', count)
            action_index += 1
            Target, scan_, pose, finish_pose, state_image = env.get_state()
            # state = a.get_state(scan_, Target).to(a.device)
            state = (a.image_tensor(state_image).unsqueeze(0), a.data_to_tensor(scan_).unsqueeze(0))
            out = a.model(state)
            #
            action = a.choose_action(out)
            print(out, action)

            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_state_image = env.step(action)

            next_state = (a.image_tensor(next_state_image), a.data_to_tensor(next_scan_))
            replay = a.buffer.write_Buffers(state, next_state, reward, action, done)

            print(reward)

            if replay is not None:
                a.train(replay)
                print('train')

            if count % 500 == 0:
                print('更新target网路')
                a.target_model.load_state_dict(a.model.state_dict())
                count = 0
            count += 1

            reward_list.append(reward)

            episode_step += 1
            if episode_step == 300:
                env.get_goalbox = True

            if env.get_goalbox:
                env.chage_finish()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                if reward_list_[-1] == max(reward_list_):
                    a.save_best()
                if a.eps > 0.05:
                    a.eps *= 0.99
                break

            if env.get_bummper:
                env.init_word()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                if reward_list_[-1] == max(reward_list_):
                    a.save_best()
                if a.eps > 0.05:
                    a.eps *= 0.9
                break

            rate.sleep()
    a.save_variable(b_list, reward_list_mean, reward_list_)
    a.save_()
    plt.plot(b_list, reward_list_mean)
    plt.show()
