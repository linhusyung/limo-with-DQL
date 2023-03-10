#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
import torch
from torch import optim
from Env import environment
from net_Lin import *
import matplotlib.pyplot as plt
import csv
import cv2
import time


class agent():
    def __init__(self, num_state, num_action, q_lr, pi_lr, target_entropy, gamma, tau, alpha_lr, imitate_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q_net1 = Q_net(num_state, num_action).to(self.device)
        self.Q_net2 = Q_net(num_state, num_action).to(self.device)

        self.Q_net1_target = Q_net(num_state, num_action).to(self.device)
        self.Q_net2_target = Q_net(num_state, num_action).to(self.device)

        self.Q_net1_target.load_state_dict(self.Q_net1_target.state_dict())
        self.Q_net2_target.load_state_dict(self.Q_net2_target.state_dict())

        self.actor = Actor_net(num_state, num_action).to(self.device)
        self.actor.load_state_dict(torch.load(imitate_path))

        self.batch_size = 64
        self.Buffers = Replay_Buffers(self.batch_size)
        self.gamma = 0.99

        self.q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.log_alpha = torch.tensor(np.log(0.05), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # self.target_entropy = target_entropy  # 目标熵的大小
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau

        self.Incremental_Learning_loss = nn.MSELoss()
        self.Incremental_Learning_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-2)

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

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def replay_resize(self, replay):
        state = torch.zeros(self.batch_size, 25).to(self.device)
        # state_scan = torch.zeros(self.batch_size, 24).to(self.device)
        # state = (state_img, state_scan)

        next_state = torch.zeros(self.batch_size, 25).to(self.device)
        # state_scan_next = torch.zeros(self.batch_size, 24).to(self.device)
        # next_state = (state_img_next, state_scan_next)

        action = torch.zeros(self.batch_size, 2, dtype=torch.float32).to(self.device)
        reward = torch.zeros(self.batch_size, 1).to(self.device)
        done = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(self.device)
        for _ in range(len(replay)):
            # state[0][_] = replay[_]['state'][0]
            # state[1][_] = replay[_]['state'][1]
            state[_] = replay[_]['state']

            # next_state[0][_] = replay[_]['next_state'][0]
            # next_state[1][_] = replay[_]['next_state'][1]
            next_state[_] = replay[_]['next_state']

            action[_] = a.np_to_tensor(replay[_]['action'])

            reward[_] = replay[_]['reward']

            done[_] = replay[_]['done']

        return state, next_state, action, reward, done

    def train(self, replay):
        print('train')
        state, next_state, action, reward, done = self.replay_resize(replay)
        action_next, log_prob = a.actor.sample(next_state)

        entropy = -log_prob
        Q1_value = self.Q_net1_target(next_state, action_next)
        Q2_value = self.Q_net2_target(next_state, action_next)
        next_value = torch.min(Q1_value, Q2_value) + self.log_alpha.exp() * entropy
        td_target = reward + self.gamma * next_value * (1 - done)

        critic_1_loss = torch.mean(F.mse_loss(self.Q_net1(state, action), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.Q_net2(state, action), td_target.detach()))

        self.q1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.q2_optimizer.step()

        # 更新actor
        # new_actions, log_prob = self.actor.sample(state)
        new_actions, log_prob = self.actor(state)
        entropy = -log_prob
        q1_value = self.Q_net1(state, new_actions)
        q2_value = self.Q_net2(state, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.Q_net1, self.Q_net1_target)
        self.soft_update(self.Q_net2, self.Q_net2_target)

    def save_variable(self, i_list, mean_reward, reward_list):
        with open('result/my.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            # 寫入一列資料
            writer.writerow(['玩的次数', i_list])
            writer.writerow(['平均奖励', mean_reward])
            writer.writerow(['奖励加总', reward_list])

    def save_(self):
        torch.save(self.actor.state_dict(), './model/sac_model_pre/my_model_params.pth')

    def Incremental_Learning(self, data):
        # print(len(data))
        state = torch.zeros(len(data), 25).to(self.device)
        next_state = torch.zeros(len(data), 25).to(self.device)
        action = torch.zeros(len(data), 2, dtype=torch.float32).to(self.device)
        reward = torch.zeros(len(data), 1).to(self.device)
        done = torch.zeros(len(data), 1, dtype=torch.float32).to(self.device)
        for _ in range(len(data)):
            state[_] = replay[_]['state']
            next_state[_] = replay[_]['next_state']
            action[_] = a.np_to_tensor(replay[_]['action'])
            reward[_] = replay[_]['reward']
            done[_] = replay[_]['done']

        x, _ = self.actor(state.to(self.device))
        loss = self.Incremental_Learning_loss(x, action.to(self.device))
        # print(x, action, loss)
        #
        self.Incremental_Learning_optimizer.zero_grad()
        loss.backward()
        self.Incremental_Learning_optimizer.step()


if __name__ == '__main__':
    pi_lr = 3e-4
    q_lr = 3e-3
    alpha_lr = 3e-3
    target_entropy = -2
    gamma = 0.99
    tau = 0.005
    num_action = 2
    num_state = 25
    b_list = []
    reward_list_ = []
    reward_list_mean = []
    imitate_path = 'model/sac_model_pre/pre_imitate.pth'

    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
    a = agent(num_state, num_action, q_lr, pi_lr, target_entropy, gamma, tau, alpha_lr, imitate_path)
    env = environment()

    done_frequency = 0
    chage_rew = True

    for i in range(1000):
        reward_list = []
        b_list.append(i)
        print('第', i, '次游戏')
        action_index = 0
        episode_step = 0
        my_text = []
        while True:
            # for i in range(10):
            print('第', action_index, '个动作')
            action_index += 1
            Target, scan_, pose, finish_pose, state_image = env.get_state()
            # state = (a.image_tensor(state_image).unsqueeze(0), a.data_to_tensor(scan_).unsqueeze(0))
            re_data = []
            for _ in range(24):
                re_data.append(scan_[_ * (len(scan_) // 24)])
            state = torch.cat(
                (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)

            action, _ = a.actor(state)
            action = a.tensor_to_numpy(action.squeeze())
            print('action', action)
            # if done_frequency >= 10:
            #     chage_rew = False
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_state_image = env.step(action,
                                                                                                            True)
            re_data_next = []
            for _ in range(24):
                re_data_next.append(next_scan_[_ * (len(next_scan_) // 24)])
            next_state = torch.cat(
                (a.data_to_tensor(next_Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data_next).unsqueeze(0)),
                1)

            # print(next_state.shape)
            episode_step += 1
            if episode_step == 200:
                env.get_bummper = True
                reward = -50
            print('reward=', reward)
            replay = a.Buffers.write_Buffers(state, next_state, reward, action, done)
            #
            if replay is not None:
                a.train(replay)

            reward_list.append(reward)

            if env.get_goalbox:
                env.chage_finish()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                done_frequency += 1
                # if reward_list_[-1] == max(reward_list_):
                #     a.save_best()
                break

            if env.get_bummper:
                env.init_word()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                done_frequency = 0
                # if reward_list_[-1] == max(reward_list_):
                #     a.save_best()
                break

            rate.sleep()

        # my_text = [i for i in replay if i['done'] != 1 and i['state'][0] != 0]
        for i in a.Buffers.buffer:
            # print(i['done'], i['state'],i['state'][0][0])
            if i['done'] != 1 and i['state'][0][0] != -1:
                # print(i)
                my_text.append(i)

        # print(my_text)
        c = sorted(my_text, key=lambda x: x['reward'])
        # print(c[-3:], len(c[-3:]))
        a.Incremental_Learning(c[-10:])
        # break

    a.save_variable(b_list, reward_list_mean, reward_list_)
    a.save_()
    plt.plot(b_list, reward_list_mean)
    plt.show()
    l1, = plt.plot(b_list, reward_list_mean)
    l2, = plt.plot(b_list, reward_list_, color='red', linewidth=1.0, linestyle='--')
    plt.legend(handles=[l1, l2], labels=['reward_mean', 'reward_sum'], loc='best')
    plt.show()
