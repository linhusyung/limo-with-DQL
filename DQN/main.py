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
        self.model = DDQN(state_dim=24 + 3)
        self.target_model = DDQN(state_dim=24 + 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1
        self.buffer = Replay_Buffers()
        self.gamma = 0.99
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float)

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

    def train(self, replay):
        print('train')
        state, next_state = [], []
        action, reward, done = [], [], []
        for _ in range(len(replay)):
            state.append(replay[_]['state'])

            next_state.append(replay[_]['next_state'])

            action.append(replay[_]['action'])
            reward.append(replay[_]['reward'])
            done.append(replay[_]['done'])

        Q = self.model(self.tuple_of_tensors_to_tensor(state)).to(self.device)
        Q_next = self.model(self.tuple_of_tensors_to_tensor(next_state)).to(self.device)
        argmaz_action = torch.argmax(Q_next, 1)

        Q_target_net = self.target_model(self.tuple_of_tensors_to_tensor(next_state)).to(self.device)
        max_Q_ = Q_target_net.gather(1, argmaz_action.unsqueeze(1).type(torch.int64)).squeeze(1)
        Q_target = torch.tensor(reward, dtype=torch.float32).to(self.device) + \
                   self.gamma * max_Q_.to(self.device) * (1 - torch.tensor(done, dtype=torch.float32).to(self.device))
        Q_eval = Q.gather(1, torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(1).type(
            torch.int64)).squeeze(1)

        loss = self.loss_fn(Q_target, Q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_variable(self, i_list, mean_reward, reward_list):
        with open('result/train_res_second.csv', 'w', newline='') as csvfile:
            # ?????? CSV ????????????
            writer = csv.writer(csvfile)

            # ??????????????????
            writer.writerow(['????????????', i_list])
            writer.writerow(['????????????', mean_reward])
            writer.writerow(['????????????', reward_list])

    def save_(self):
        torch.save(self.model.state_dict(), 'model/model_params_third.pth')
    def save_best(self):
        print('???????????????model')
        torch.save(self.model.state_dict(), 'model/model_params_best_third.pth')


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
        print('???', i, '?????????')
        print(a.eps)
        episode_step = 0
        action_index = 0
        while True:
            print('???', action_index, '?????????')
            print('count', count)
            action_index += 1
            Target, scan_, pose, finish_pose = env.get_state()
            state = a.get_state(scan_, Target)

            out = a.model(state).to(a.device)
            action = a.choose_action(out)
            print(out, action)
            #
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done = env.step(action)
            next_state = a.get_state(next_scan_, next_Target)
            replay = a.buffer.write_Buffers(state, next_state, reward, action, done)
            #
            print(reward)
            #
            if replay is not None:
                a.train(replay)

            if count % 500 == 0:
                print('??????target??????')
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
                if reward_list_[-1]==max(reward_list_):
                    a.save_best()
                if a.eps > 0.05:
                    a.eps *= 0.99
                break
            if env.get_bummper:
                env.init_word()
                print('reward_list', sum(reward_list))
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                if reward_list_[-1]==max(reward_list_):
                    a.save_best()
                if a.eps > 0.05:
                    a.eps *= 0.9
                break

            rate.sleep()
    a.save_variable(b_list, reward_list_mean, reward_list_)
    a.save_()
    plt.plot(b_list, reward_list_mean)
    plt.show()
