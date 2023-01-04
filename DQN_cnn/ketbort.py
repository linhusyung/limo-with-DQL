import rospy


if __name__ == '__main__':
    rospy.init_node('action_key', anonymous=True)
    rate = rospy.Rate(1)
    while True:
        action=int(input('输入动作：'))
        rospy.set_param('/key', action)
        rate.sleep()