import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from imitate_networl import *


class My_dataset(Dataset):
    def __init__(self, image_path, scan_path, target_path):
        self.image_path = [image_path + x for x in os.listdir(r"" + image_path)]
        self.scan_path = [scan_path + x for x in os.listdir(r"" + scan_path)]
        self.target_path = [target_path + x for x in os.listdir(r"" + target_path)]

    def __getitem__(self, index):
        image = np.load(self.image_path[index])
        image_Target = np.expand_dims(image, axis=0)

        scan = np.load(self.scan_path[index])
        re_data = []
        for _ in range(24):
            re_data.append(scan[_ * (len(scan) // 24)])
        data = np.concatenate((image_Target, re_data), axis=0)

        target = np.load(self.target_path[index])
        target = target[0]
        target[0], target[1] = target[1], target[0]
        return data, target

    def __len__(self):
        return len(self.scan_path)


class imitate_learning():
    def __init__(self, image_path, scan_path, target_path, image_test_path, scan_test_path, target_test_path,
                 model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.my = My_dataset(image_path, scan_path, target_path)
        self.train_loader = DataLoader(dataset=self.my, batch_size=8, shuffle=True)

        self.test = My_dataset(image_test_path, scan_test_path, target_test_path)
        self.test_loader = DataLoader(dataset=self.test, batch_size=1, shuffle=False)

        self.actor = Actor_net(25, 2).to(self.device)
        self.actor.load_state_dict(torch.load(model_path))

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.epoch = 10

    def train(self):
        for batch, (data, label) in enumerate(self.train_loader):
            action, log_prob = self.actor(data.float().to(self.device))

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
        torch.save(self.actor.state_dict(), './result/2_3/model_params_fine_tuning.pth')

    def step(self):
        for epoch in range(self.epoch):
            print('epoch', epoch)
            self.train()
        self.test_()
        self.save_()


if __name__ == '__main__':
    image_path = './data/image_target/'
    scan_path = './data/scan/'
    target_path = './data/target/'

    image_test_path = './data/image_target_test/'
    scan_test_path = './data/scan_test/'
    target_test_path = './data/target_test/'

    model_path = 'model/sac_model/model_params_2_3.pth'

    imitate = imitate_learning(image_path, scan_path, target_path, image_test_path, scan_test_path, target_test_path,
                               model_path)
    imitate.step()
    # imitate.test_()