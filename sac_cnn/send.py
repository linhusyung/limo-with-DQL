#!/usr/bin/env python
# license removed for brevity
import numpy as np
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from Env import environment
# from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
import cv2
import csv


def movebase_client(finish_pose):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = finish_pose[0]
    goal.target_pose.pose.position.y = finish_pose[1]
    goal.target_pose.pose.orientation.z = 0
    goal.target_pose.pose.orientation.w = 1

    client.send_goal(goal)

    # client.wait_for_result()


class sub():
    def __init__(self):
        self.x = 0
        self.z = 0

    def get_target_vel(self, data):
        self.x = data.linear.x
        self.z = data.angular.z


def save_variable(i_list, mean_reward, reward_list):
    with open('result/2_3/navigation_test.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入一列資料
        writer.writerow(['玩的次数', i_list])
        writer.writerow(['平均奖励', mean_reward])
        writer.writerow(['奖励加总', reward_list])


if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    env = environment()
    rate = rospy.Rate(30)
    sub = sub()
    # b = 0
    # c = 0
    epoch_list = []
    reward_list_ = []
    reward_list_mean = []
    for i in range(10):
        epoch_list.append(i)
        reward_list = []
        episode_step = 0
        while True:
            movebase_client(env.finish_pose)
            Target, scan_, pose, finish_pose, state_image = env.get_state()
            cmd_vel = rospy.Subscriber("cmd_vel", Twist, sub.get_target_vel)
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_state_image = env.step(0,
                                                                                                            True)
            print(reward)
            # if Target != -1:
            #     """
            #     target=[[sub.x(action.v),sub.z(action.w)]
            #             [reward,0]]
            #     """
            #     target = np.zeros((2, 2))
            #     # print(sub.x, sub.z)
            #     target[0][0] = sub.x
            #     target[0][1] = sub.z
            #     target[1][0] = reward
            #
            #     # cv2.imwrite('./data/image/image_' + str(b) + '.jpg', state_image)
            #
            #     np.save('./data/image_target_test/image_target_' + str(b) + '.npy', Target)
            #     np.save('./data/scan_test/scan_' + str(b) + '.npy', scan_)
            #     np.save('./data/target_test/target_' + str(b) + '.npy', target)
            #
            #     # cv2.imwrite('./data/next_image/next_state_image' + str(b) + '.jpg', state_image)
            #     # np.save('./data/next_scan/scan_' + str(b) + '.npy', scan_)
            #     # np.save('./data/next_scan/scan_' + str(b) + '.npy', scan_)
            #
            #     cv2.imshow('img', state_image)
            #     cv2.waitKey(1)
            #     print('b',b)
            #     b += 1
            # print(c)
            # c += 1
            episode_step += 1
            print(episode_step)
            if episode_step == 50:
                env.get_bummper = True
                reward = -50
            reward_list.append(reward)
            if env.get_goalbox:
                env.chage_finish()
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                break

            if env.get_bummper:
                env.chage_finish()
                reward_list_.append(sum(reward_list))
                reward_list_mean.append(np.mean(reward_list_))
                break
            rate.sleep()
        save_variable(epoch_list, reward_list_mean, reward_list_)
