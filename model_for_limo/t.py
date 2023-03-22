import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pynput import keyboard
import torch
import time


class img():
    def __init__(self):
        self.bridge = CvBridge()
        self.train_index = 0
        self.text_index = 0
        self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/cap_data_tiny/best.pt', source='local',
                                     force_reload=False)
        self.yolov7.eval()
        self.last_time = time.time()
        self.fps_count = 0

        self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        # self.sub_image = rospy.Subscriber('/limo/color/image_raw', Image, self.get_image)

    def get_image(self, image):
        cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        # print(cv_im.shape, type(cv_im))
        with torch.no_grad():
            results = self.yolov7(cv_im)
            bboxes = np.array(results.pandas().xyxy[0])
        print(bboxes)
        for i in bboxes:
            if i[4] > 0:
                x_f = (i[2] - i[0]) / 2 + i[0]
                y_f = (i[3] - i[1]) / 2 + i[1]
                self.bboxes_show(cv_im, i, (int(x_f), int(y_f)))

        cv2.imshow('img', cv_im)
        cv2.waitKey(1)

        if (time.time() - self.last_time >= 1):
            print(self.fps_count)
            self.fps_count = 0
            self.last_time = time.time()

        self.fps_count += 1

    def bboxes_show(self, img, bbox, midpoint):
        cv2.circle(img, midpoint, 1, (0, 0, 255), 4)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 4)
        cv2.putText(img, bbox[6], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(bbox[4]), (int(bbox[2]), int(bbox[3]) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    i = img()
    rospy.spin()
