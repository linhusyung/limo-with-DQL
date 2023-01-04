#!/home/a/anaconda3/envs/torch/bin/python3

import rospy
import numpy as np
from Env import environment
import cv2
from network import *


class data():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DDQN(state_dim=24 + 3)
        self.PATH = 'model/model_params.pth'

    def endl(self):
        if env.get_goalbox:
            env.chage_finish()
        if env.get_bummper:
            env.init_word()
        rate.sleep()

    def choose_action(self, out: torch.tensor) -> int:
        return int(out.argmax().cpu().numpy())

    def get_state(self, scan_, taget) -> torch.tensor:
        # pose_finish_pose = torch.cat((self.data_to_tensor(pose), self.data_to_tensor(finish_pose)), -1)
        return torch.cat((self.data_to_tensor(scan_), self.data_to_tensor(taget)), -1)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float)

    def save(self, i, image, now: bool):
        if now:
            cv2.imwrite('data/' + str(i) + '.jpg', image)
        else:
            cv2.imwrite('data_next/' + str(i) + '_next.jpg', image)

    def list_to_np(self, data):
        return np.array(data)

    def int_to_np(self, data):
        return np.expand_dims(np.array(data), axis=1)

def main():
    d.model.load_state_dict(torch.load(d.PATH))
    Epoch = 500

    Target_list = []
    Target_next_list = []

    scan_list = []
    scan_next_list = []

    action_list = []
    reward_list = []

    done_list = []

    for i in range(Epoch):
        Target, scan_, pose, finish_pose, state_image = env.get_state()
        d.save(i, state_image, now=True)

        Target_list.append(Target)
        scan_list.append(scan_)

        cv2.imshow('mask', state_image)
        cv2.waitKey(1)

        state = d.get_state(scan_, Target)
        out = d.model(state).to(d.device)
        action = d.choose_action(out)
        print(out, action)

        action_list.append(action)

        next_Target, next_scan_, next_pose, next_finish_pose, reward, done, state_image_next = env.step(action)

        Target_next_list.append(next_Target)
        scan_next_list.append(next_scan_)
        reward_list.append(reward)
        done_list.append(done)

        d.save(i, state_image_next, now=False)
        print(reward, done)

        d.endl()
        rate.sleep()
    Target_ = d.list_to_np(Target_list)
    np.save('target/Target.npy',Target_)
    Target_next_ = d.list_to_np(Target_list)
    np.save('target/Target_next_.npy', Target_next_)

    scan_list_ = d.list_to_np(scan_list)
    np.save('target/scan_list_.npy', scan_list_)
    scan_next_list_ = d.list_to_np(scan_next_list)
    np.save('target/scan_next_list_.npy', scan_next_list_)

    action_ = d.int_to_np(action_list)
    np.save('target/action_.npy', action_)
    reward_ = d.int_to_np(reward_list)
    np.save('target/reward.npy', reward_)

    done_ = d.int_to_np(done_list)
    np.save('target/done_.npy', done_)


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(1)
    d = data()
    env = environment()
    main()
