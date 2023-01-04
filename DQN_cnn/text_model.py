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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PATH_cnn = 'model/model_cnn.pth'
        self.PATH_Linear = 'model/model_Linear.pth'
        self.model_cnn=DDQN_cnn().to(self.device)
        self.model_Linear=DDQN_Linear(state_dim=24 + 3).to(self.device)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def choose_action(self, out: torch.tensor) -> int:
        return int(out.argmax().cpu().numpy())

    def image_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 1, 0))).float().to(self.device)

if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
    a = agent()
    env = environment()
    a.model_cnn.load_state_dict(torch.load(a.PATH_cnn))
    a.model_Linear.load_state_dict(torch.load(a.PATH_Linear))
    for i in range(10):
        print('第', i, '次游戏')
        while True:
            Target, scan_, pose, finish_pose, state_image = env.get_state()
            image_input=a.image_tensor(state_image).unsqueeze(0)
            scan_input=a.data_to_tensor(scan_).unsqueeze(0)

            cnn_out=a.model_cnn(image_input)
            out=a.model_Linear(scan_input,cnn_out)
            print('cnn_out',cnn_out)
            print('out',out)
            action = a.choose_action(out)


            env.step(action)

            if env.get_goalbox:
                env.chage_finish()
                break
            if env.get_bummper:
                env.init_word()
                break

            rate.sleep()