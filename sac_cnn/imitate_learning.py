import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
# from imitate_networl import *
from net_Lin import *


class My_dataset(Dataset):
    def __init__(self, path):
        # target_path = './Expert_data/target'
        # scan_path = './Expert_data/scan'
        # heading_path = './Expert_data/heading'
        #
        # action_path = './Expert_data/action'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_path = [path[0] + x for x in os.listdir(r"" + path[0])]
        self.scan_path = [path[1] + x for x in os.listdir(r"" + path[1])]
        self.heading_path = [path[2] + x for x in os.listdir(r"" + path[2])]

        self.action_path = [path[3] + x for x in os.listdir(r"" + path[3])]

    def __getitem__(self, index):
        target = np.load(self.target_path[index])
        target = self.np_to_tensor(target).unsqueeze(0).unsqueeze(0)

        scan = np.load(self.scan_path[index])
        re_data = []
        for _ in range(24):
            re_data.append(scan[_ * (len(scan) // 24)])
        re_data = self.data_to_tensor(re_data).unsqueeze(0)

        heading = np.load(self.heading_path[index])
        heading = self.np_to_tensor(heading).unsqueeze(0).unsqueeze(0)

        state = torch.cat(
            (target, re_data, heading), 1)

        action = np.load(self.action_path[index])
        action = action[0]
        action[0], action[1] = action[1], action[0]
        action = self.np_to_tensor(action).unsqueeze(0)
        return torch.squeeze(state), torch.squeeze(action)

    def __len__(self):
        return len(self.target_path)

    def np_to_tensor(self, data):
        return torch.from_numpy(data).float().to(self.device)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)


class imitate_learning():
    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.my = My_dataset(path)
        self.train_loader = DataLoader(dataset=self.my, batch_size=8, shuffle=True)

        # self.test = My_dataset(path)
        # self.test_loader = DataLoader(dataset=self.test, batch_size=1, shuffle=False)

        self.actor = Actor_net(26, 2).to(self.device)
        # self.actor.load_state_dict(torch.load(model_path))

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.epoch = 100

    def train(self):
        for batch, (data, label) in enumerate(self.train_loader):
            action, log_prob = self.actor(data)
            loss = self.loss(action, label.float().to(self.device))
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 14 == 0:
                loss, current = loss.item(), batch * len(data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.my):>5d}]")

    def test_(self):
        loss_test_avg = 0
        self.actor.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                action, _ = self.actor(X.float().to(self.device))
                loss_test = self.loss(action, y.float().to(self.device))
                loss_test_avg += loss_test
                print(loss_test)
            loss_test_avg /= len(self.test)
        print('loss_test_avg:', loss_test_avg)

    def save_(self):
        torch.save(self.actor.state_dict(), './Expert_data/pre_.pth')

    def step(self):
        for epoch in range(self.epoch):
            print('epoch', epoch)
            self.train()
        # self.test_()
        # self.save_()


if __name__ == '__main__':
    target_path = './Expert_data/target/'
    scan_path = './Expert_data/scan/'
    heading_path = './Expert_data/heading/'

    action_path = './Expert_data/action/'

    path = [target_path, scan_path, heading_path, action_path]
    # model_path = 'model/sac_model/model_params_2_3.pth'

    im = imitate_learning(path)
    im.step()
    im.save_()
    # imitate.test_()
