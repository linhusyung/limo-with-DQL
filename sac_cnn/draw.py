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
    # print(reward_navigation_acml)

    path_navigation = 'result/2_3/sac_imitate_test.csv'
    k_2 = d.open(path_navigation)
    _, _, reward_sac_imitate = d.read(k_2)
    # print(reward_navigation_acml)


    '''
    直方图
    '''
    plt.title("test")  # 圖的標題
    plt.xlabel("name")  # x軸的名稱
    plt.ylabel("mean_reward")  # y軸的名稱
    x = ['sac','navigation','navigation_acml','sac_imitate']
    y = [np.mean(reward_sac),np.mean(reward_navigation),np.mean(reward_navigation_acml),np.mean(reward_sac_imitate)]
    sns.set(style="whitegrid")
    plt.bar(x, y)  # 繪製長條圖
    plt.show()  # 顯現圖形
    '''
    '''
    # path_sac = 'result/2_3/sac_reward.csv'
    # d = draw()
    # k = d.open(path_sac)
    # batch, mean_reward, reward_sac = d.read(k)
    #
    # path_sac_imitate = 'result/2_3/imitate_init_reward.csv'
    # k_1 = d.open(path_sac_imitate)
    # batch_imitate, mean_reward_imitate, reward_sac_imitate = d.read(k_1)

    # print(len(mean_reward),len(mean_reward_imitate))
    '''
    折线图
    '''
    # cars_df = pd.DataFrame(
    #     {"sac": mean_reward,
    #      "sac_imitate": mean_reward_imitate
    #      }
    # )
    #
    # sns.lineplot(data = cars_df)
    # plt.show()
