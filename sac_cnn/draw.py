import csv
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
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
    # path_sac = 'result/2_3/reward/sac_test.csv'
    # d = draw()
    # k = d.open(path_sac)
    # _, _, reward_sac = d.read(k)
    #
    # path_navigation = 'result/2_3/reward/navigation_test.csv'
    # k_1 = d.open(path_navigation)
    # _, _, reward_navigation = d.read(k_1)
    #
    # path_navigation = 'result/2_3/reward/navigation_acml_test.csv'
    # k_2 = d.open(path_navigation)
    # _, _, reward_navigation_acml = d.read(k_2)
    # # print(reward_navigation_acml)
    #
    # path_navigation = 'result/2_3/reward/sac_imitate_test.csv'
    # k_3 = d.open(path_navigation)
    # _, _, reward_sac_imitate = d.read(k_3)
    # # print(reward_navigation_acml)
    #
    # path_sac_navigation = 'result/2_3/reward/navigation_with_sac_imitate_test.csv'
    # k_4 = d.open(path_navigation)
    # _, _, reward_sac_imitate_navigation = d.read(k_4)
    #
    # path_pre = 'result/2_3/pre.csv'
    # k_pre = d.open(path_pre)
    # _, _, reward_pre = d.read(k_pre)

    '''
    直方图
    '''
    # plt.title("test")  # 圖的標題
    # plt.xlabel("name")  # x軸的名稱
    # plt.ylabel("mean_reward")  # y軸的名稱
    # # x = ['sac','navigation','navigation acml','sac imitate','navigation_sac','pre']
    # x = ['sac', 'navigation', 'navigation acml', 'sac imitate', 'pre']
    # y = [np.mean(reward_sac), np.mean(reward_navigation), np.mean(reward_navigation_acml), np.mean(reward_sac_imitate),
    #      np.mean(reward_pre)]
    # sns.set(style="whitegrid")
    # plt.bar(x, y)  # 繪製長條圖
    # plt.show()  # 顯現圖形
    '''
    '''
    path_sac = 'result/2_3/sac/sac_reward.csv'
    # path_sac = 'result/sac_pose.csv'
    # path_sac = 'result/2_3/sac_pose/scan50/sac_pose_1.csv'
    d = draw()
    k = d.open(path_sac)
    batch, mean_reward, reward_sac = d.read(k)

    path_sac_2 = 'result/2_3/sac/sac_reward_1.csv'
    # path_sac_2 = 'result/2_3/sac_pose/scan50/sac_pose_scan50_1.csv'
    k_0 = d.open(path_sac_2)
    batch_sac_2, mean_reward_sac_2, reward_sac_2 = d.read(k_0)

    # sac_mean_reward = np.vstack((mean_reward, mean_reward_sac_2))
    sac_mean_reward = np.vstack((reward_sac, reward_sac_2))


    # path_sac_imitate = 'result/2_3/sac_imitate_model/imitate_init_reward.csv'
    path_sac_imitate = 'result/2_3/sac_imitate_model/imitate_1.csv'
    k_1 = d.open(path_sac_imitate)
    batch_imitate, mean_reward_imitate, reward_sac_imitate = d.read(k_1)

    path_sac_imitate = 'result/2_3/sac_imitate_model/imitate_init_reward.csv'
    k_2 = d.open(path_sac_imitate)
    batch_sac, mean_reward_sac, reward_sac_sac = d.read(k_2)

    # rl_mean_reward = np.vstack((mean_reward_imitate, mean_reward_sac))
    rl_mean_reward = np.vstack((reward_sac_imitate, reward_sac_sac))
    print(len(rl_mean_reward), rl_mean_reward.shape)


    '''
    折线图
    '''
    data = (sac_mean_reward, rl_mean_reward)
    label = ['sac', 'sac with imitate']
    df = []
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='reward'))
        df[i]['method'] = label[i]

    df = pd.concat(df)
    print(df)
    sns.lineplot(x="episode", y="reward", hue="method", style="method", data=df)
    plt.show()
    '''
    '''
    sac_mean_reward_ = np.concatenate((mean_reward, mean_reward_sac_2))
    episode1 = range(len(reward_sac))
    episode2 = range(len(reward_sac_2))
    episode = np.concatenate((episode1, episode2))

    sns.lineplot(x=episode, y=sac_mean_reward_)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

