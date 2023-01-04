import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    path_1 = 'result/train_res_frist.csv'
    path_2 = 'result/train_res_second.csv'
    path_3 = 'result/train_res_third.csv'
    path_list = [path_1, path_2, path_3]
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

    sns.set()
    rewards = np.concatenate((mean_reward_[0], mean_reward_[1], mean_reward_[2]))  # 合并数组
    episode1 = range(len(mean_reward_[0]))
    episode2 = range(len(mean_reward_[1]))
    episode3 = range(len(mean_reward_[2]))
    episode = np.concatenate((episode1, episode2, episode2))
    sns.lineplot(x=episode, y=rewards)
    plt.xlabel("episode")
    plt.ylabel("mean_reward")
    plt.show()
