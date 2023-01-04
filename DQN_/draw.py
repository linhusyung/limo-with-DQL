import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
k=[]
k_1=[]
with open('result/train_res.csv', newline='') as csvfile:

    rows = csv.reader(csvfile)
    for i in rows:
        k.append(i)

batch=k[0]
mean_reward=k[1]
reward=k[2]
mean_reward=eval(mean_reward[1])
batch=eval(batch[1])
reward=eval(reward[1])
print(min(mean_reward),max(mean_reward),min(reward),max(reward))

# print(batch[1])
plt.plot(batch,reward)
plt.show()