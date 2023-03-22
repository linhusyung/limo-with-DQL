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


def batch_read(local_path):
    batch_list = []
    mean_reward_list = []
    reward_sac_list = []
    d = draw()
    for i in local_path:
        batch, mean_reward, reward_sac = d.read(d.open(i))
        batch_list.append(batch)
        mean_reward_list.append(mean_reward)
        reward_sac_list.append(reward_sac)
    batch = np.vstack((batch_list[0], batch_list[1]))
    mean_reward = np.vstack((mean_reward_list[0], mean_reward_list[1]))
    reward = np.vstack((reward_sac_list[0], reward_sac_list[1]))
    return batch, mean_reward, reward


def np_to_pd(data):
    label = ['sac 24', 'sac 50', 'sac 100']
    df = []
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='mean_reward'))
        df[i]['method'] = label[i]
    df = pd.concat(df)
    return df


def main(path: list):
    data = []
    for _ in path:
        batch, mean_reward, reward = batch_read(_)
        data.append(mean_reward)
    df = np_to_pd(data)
    print(df)
    sns.lineplot(x="episode", y="mean_reward", hue="method", style="method", data=df)
    plt.show()


if __name__ == '__main__':
    path_sac_24_1 = 'result/map_3world/scan_24/sac_1.csv'
    path_sac_24_2 = 'result/map_3world/scan_24/sac_2.csv'

    path_sac_50_1 = 'result/map_3world/scan_50/sac_1.csv'
    path_sac_50_2 = 'result/map_3world/scan_50/sac_2.csv'

    path_sac_100_1 = 'result/map_3world/scan_100/sac_1.csv'
    path_sac_100_2 = 'result/map_3world/scan_100/sac_2.csv'

    path_24 = [path_sac_24_1, path_sac_24_2]
    path_50 = [path_sac_50_1, path_sac_50_2]
    path_100 = [path_sac_100_1, path_sac_100_2]
    #
    path = [path_24, path_50, path_100]
    # path = [path_24]
    main(path)
