#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
from Env import environment

if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(50)
    env = environment()
    # for i in range(50):
    while True:
        Target, scan_, pose, finish_pose, state_image = env.get_state()
        print(Target)
        rate.sleep()