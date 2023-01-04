#!/home/a/anaconda3/envs/torch/bin/python3
import torch
from collections import deque
from random import sample
import torch.nn as nn
import torch.nn.functional as F


class DDQN_cnn(nn.Module):
    def __init__(self):
        super(DDQN_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=5)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5)

        self.fc1 = nn.Linear(5832, 648)
        self.fc2 = nn.Linear(648, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool_1(out)
        #
        out = self.conv2(out)
        out = self.pool_2(out)

        out = self.conv3(out)

        out = torch.flatten(out, 1)
        #
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out


class DDQN_Linear(nn.Module):
    def __init__(self, state_dim):
        super(DDQN_Linear, self).__init__()
        self.h1 = nn.Linear(state_dim, 512)
        self.h1.weight.data.normal_(0, 0.1)
        self.h2 = nn.Linear(512, 256)
        self.h2.weight.data.normal_(0, 0.1)
        self.h3 = nn.Linear(256, 64)
        self.h3.weight.data.normal_(0, 0.1)
        self.h4 = nn.Linear(64, 5)
        self.h4.weight.data.normal_(0, 0.1)

    def forward(self, x, y):
        x_y = torch.cat((x, y),1)
        out = F.relu(self.h1(x_y))
        out = F.relu(self.h2(out))
        out = F.relu(self.h3(out))
        out = self.h4(out)
        return out


class DDQN_Global(nn.Module):
    def __init__(self):
        super(DDQN_Global, self).__init__()
        self.cnn = DDQN_cnn()
        self.Linear = DDQN_Linear(state_dim=24 + 3)

    def forward(self, state: tuple):
        out_cnn = self.cnn(state[0])
        out = self.Linear(state[1], out_cnn)
        return out


class Replay_Buffers():
    def __init__(self, batch_size):
        self.buffer_size = 10000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = batch_size

    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done, }
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)


# import rospy
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import Image
# from main import *
# if __name__ == '__main__':
#     rospy.init_node('text_listener', anonymous=True)
#     PATH = 'model/model_cnn.pth'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cnn=DDQN_cnn().to(device)
#     cnn.load_state_dict(torch.load(PATH))
#     bridge = CvBridge()
#     a=agent()
#     rate = rospy.Rate(50)
#     for _ in range(10):
#         im = bridge.imgmsg_to_cv2(rospy.wait_for_message('/limo/color/image_raw', Image), 'bgr8')
#         state_image = cv2.resize(im, (100, 100))
#         image_tensor=a.image_tensor(state_image).unsqueeze(0)
#         out=cnn(image_tensor)
#         print(out)
#         rate.sleep()