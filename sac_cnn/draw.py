import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class draw():
    def __init__(self):
        pass

    def open(self, path: str) -> list:
        k = []
        with open(path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for i in rows:
                k.append(i)
        return k

    def read(self, k):
        batch = k[0]
        mean_reward = k[1]
        reward = k[2]
        batch = eval(batch[1])
        mean_reward = eval(mean_reward[1])
        reward = eval(reward[1])
        return batch, mean_reward, reward


if __name__ == '__main__':
    path_1 = 'result/sac.csv'
    path_2 = 'result/sac_2.csv'
    path_3 = 'result/sac_3.csv'
    path_4 = 'result/sac_cnn.csv'
    path_5 = 'result/sac_cnn_1.csv'
    path_6 = 'result/sac_cnn.csv_2'
    path_list = [path_1, path_2, path_3, path_4, path_5, path_6]
    d = draw()
    batch_ = []
    mean_reward_ = []
    reward_ = []
    for i in path_list:
        k = d.open(i)
        batch, mean_reward, reward = d.read(k)
        batch_.append(batch)
        mean_reward_.append(mean_reward)
        reward_.append(reward)
    # k = d.open(path_1)
    # batch, mean_reward, reward = d.read(k)
    # # l1, = plt.plot(batch, mean_reward)
    # l2, = plt.plot(batch, mean_reward, color='red', linewidth=1.0, linestyle='--')
    # # plt.legend(handles=[l1, l2], labels=['reward_mean', 'reward'], loc='best')
    # plt.show()
    sns.set()
    sac = np.concatenate((mean_reward_[0], mean_reward_[1], mean_reward_[2]))  # 合并数组
    sac_cnn = np.concatenate((mean_reward_[3], mean_reward_[4], mean_reward_[5]))  # 合并数组

    rewards = np.concatenate((reward_[0], reward_[1], reward_[2]))  # 合并数组
    # print(rewards)
    episode1 = range(len(mean_reward_[0]))
    episode2 = range(len(mean_reward_[1]))
    episode3 = range(len(mean_reward_[2]))
    episode = np.concatenate((episode1, episode2, episode3))
    y = (sac, sac_cnn)

    sns.lineplot(x=episode, y=sac_cnn)
    plt.xlabel("episode")
    plt.ylabel("mean_reward")
    plt.show()
