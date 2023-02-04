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

    def read_navigation_test(self, k):
        batch = k[0]
        reward = k[1]

        batch = eval(batch[1])
        reward = eval(reward[1])

        return batch, reward


if __name__ == '__main__':
    path_sac = 'result/2_3/sac_test.csv'
    d = draw()
    k = d.open(path_sac)
    _, _, reward_sac = d.read(k)

    path_navigation = 'result/2_3/navigation_test.csv'
    k_1 = d.open(path_navigation)
    _, _, reward_navigation = d.read(k_1)

    path_navigation = 'result/2_3/navigation_acml_test.csv'
    k_2 = d.open(path_navigation)
    _, _, reward_navigation_acml = d.read(k_2)
    print(reward_navigation_acml)
    # batch, reward = d.read_navigation_test(k)
    # print(reward)
    '''
    折线图
    '''
    # cars_df = pd.DataFrame(
    #     {"mean_reward": mean_reward,
    #      "epoch": batch
    #      }
    # )
    #
    # sns.lineplot(x = "epoch", y = "mean_reward", data = cars_df)
    # plt.show()
    '''
    直方图
    '''
    plt.title("test")  # 圖的標題
    plt.xlabel("name")  # x軸的名稱
    plt.ylabel("mean_reward")  # y軸的名稱
    x = ['sac','navigation','navigation_acml']
    y = [np.mean(reward_sac),np.mean(reward_navigation),np.mean(reward_navigation_acml)]
    sns.set(style="whitegrid")
    plt.bar(x, y)  # 繪製長條圖
    plt.show()  # 顯現圖形

    # path_2 = 'result/sac_2.csv'
    # path_3 = 'result/sac_3.csv'
    # path_4 = 'result/sac_cnn.csv'
    # path_5 = 'result/sac_cnn_1.csv'
    # path_6 = 'result/sac_cnn.csv_2'
    # path_list = [path_1, path_2, path_3, path_4, path_5, path_6]
    # d = draw()
    # batch_ = []
    # mean_reward_ = []
    # reward_ = []
    # for i in path_list:
    #     k = d.open(i)
    #     batch, mean_reward, reward = d.read(k)
    #     batch_.append(batch)
    #     mean_reward_.append(mean_reward)
    #     reward_.append(reward)
    # k = d.open(path_1)
    # batch, mean_reward, reward = d.read(k)
    # # l1, = plt.plot(batch, mean_reward)
    # l2, = plt.plot(batch, mean_reward, color='red', linewidth=1.0, linestyle='--')
    # # plt.legend(handles=[l1, l2], labels=['reward_mean', 'reward'], loc='best')
    # plt.show()

    # sns.set()
    # sac = np.concatenate((mean_reward_[0], mean_reward_[1], mean_reward_[2]))  # 合并数组
    # sac_cnn = np.concatenate((mean_reward_[3], mean_reward_[4], mean_reward_[5]))  # 合并数组
    #
    # rewards = np.concatenate((reward_[0], reward_[1], reward_[2]))  # 合并数组
    # # print(rewards)
    # episode1 = range(len(mean_reward_[0]))
    # episode2 = range(len(mean_reward_[1]))
    # episode3 = range(len(mean_reward_[2]))
    # episode = np.concatenate((episode1, episode2, episode3))
    # y = (sac, sac_cnn)
    #
    # sns.lineplot(x=episode, y=sac_cnn)
    # plt.xlabel("episode")
    # plt.ylabel("mean_reward")
    # plt.show()
