#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
from Env import environment
from net_Lin import *
import random
import cv2


class agent():
    def __init__(self, num_state, num_action, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor_net(num_state, num_action).to(self.device)
        self.actor.load_state_dict(torch.load(path))
        self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/my_data_tiny/best.pt', source='local',
                                     force_reload=False)
        self.yolov7.eval()

    def get_state(self, scan_, taget) -> torch.tensor:
        # pose_finish_pose = torch.cat((self.data_to_tensor(pose), self.data_to_tensor(finish_pose)), -1)
        return torch.cat((self.data_to_tensor(scan_), self.data_to_tensor(taget)), -1)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def image_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 1, 0))).float().to(self.device)

    def tensor_to_numpy(self, data):
        return data.detach().cpu().numpy()

    def np_to_tensor(self, data):
        return torch.from_numpy(data).float().to(self.device)

    def resize_scan(self, scan):
        # print(len(list(set(scan))))
        # temp = []
        # x = len(list(set(scan))) // 24
        # for i in range(24):
        #     temp.append(list(set(scan))[i * x:(i + 1) * x:])
        #
        # print(temp)
        # print(len(temp))
        return random.sample(list(set(scan)), 24)

    def bboxes_show(self, img, bbox, midpoint):
        cv2.circle(img, midpoint, 1, (0, 0, 255), 4)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 4)
        cv2.putText(img, bbox[6], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(bbox[4]), (int(bbox[2]), int(bbox[3]) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)

    def yolo_bbox_midpoint(self, im_x, midpoint):
        if midpoint[0] is None:
            return [0, 0, 0]
        if midpoint[0] <= im_x.shape[1] // 3:
            return [1, 0, 0]
        if midpoint[0] > im_x.shape[1] // 3 and midpoint[0] <= im_x.shape[1] // 3 * 2:
            return [0, 1, 0]
        else:
            return [0, 0, 1]


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
    num_action = 2
    num_state = 24 + 3
    path = 'model/model_params.pth'
    # path='model/model_best_.pth'
    a = agent(num_state, num_action, path)
    env = environment()

    # for i in range(10):
    action_index = 0
    while True:
        # for i in range(200):
        print('第', action_index, '个动作')
        action_index += 1
        (x_f, y_f) = (None, None)
        Target, scan_, state_image = env.get_state()
        scan = a.resize_scan(scan_)
        with torch.no_grad():
            results = a.yolov7(state_image)
            bboxes = np.array(results.pandas().xyxy[0])
        for i in bboxes:
            if i[4] > 0.5:
                x_f = (i[2] - i[0]) / 2 + i[0]
                y_f = (i[3] - i[1]) / 2 + i[1]
                a.bboxes_show(state_image, i, (int(x_f), int(y_f)))
        cv2.imshow('img', state_image)
        cv2.waitKey(1)
        Target = a.yolo_bbox_midpoint(state_image, (x_f, y_f))
        print(Target)
        state = torch.cat((a.data_to_tensor(Target).unsqueeze(0), a.data_to_tensor(scan).unsqueeze(0)), 1)
        action, _ = a.actor(state)

        # print(Target)
        action = a.tensor_to_numpy(action.squeeze())
        print('action', action)
        # next_Target, next_scan_, next_state_image = env.step(action)
        rate.sleep()
