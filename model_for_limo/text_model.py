#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
from Env import environment
from network import *
import matplotlib.pyplot as plt
import csv
import cv2
import random


class agent():
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DDQN_Global().to(self.device)
        self.PATH = 'model/model_Global.pth'

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def image_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 1, 0))).float().to(self.device)

    def choose_action(self, out: torch.tensor) -> int:
        return int(out.argmax().cpu().numpy())

    def resize_scan(self, scan):
        # print(len(list(set(scan))))
        temp = []
        x = len(scan) // 24
        for i in range(24):
            temp.append((scan)[i * x:(i + 1) * x:])
        # print(len(temp))
        choose_scan = []
        for i in temp:
            # print(i)
            c = 0
            while True:
                a = random.sample(list(i), 1)
                if a[0] != 0:
                    choose_scan.append(a[0])
                    break
                elif c > 24:
                    choose_scan.append(a[0])
                    break
                c += 1
        # return random.sample(list(set(scan)),24)
        return choose_scan


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
    a = agent()
    env = environment()
    a.model.load_state_dict(torch.load(a.PATH))
    Target_ = [0, 1, 0]
    # while True:
    for i in range(1000):
        Target, scan_, image = env.get_state()

        scan = a.resize_scan(scan_)
        print(scan, len(scan))
        state = (a.image_tensor(image).unsqueeze(0), a.data_to_tensor(scan).unsqueeze(0))


        out = a.model(state)
        action = a.choose_action(out)

        env.step(action)

        rate.sleep()
