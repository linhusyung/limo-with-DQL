#!/home/a/anaconda3/envs/torch/bin/python3
from nav_msgs.msg import Odometry
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan, Image, Imu


class odom():
    def __init__(self):
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        # self.sub_pose = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.sub_imu = rospy.Subscriber('/limo/imu', Imu, self.get_imu)
        self.c = 0

    def getOdometry(self, odom):
        # print('odom')
        self.odom = odom

    def get_imu(self,imu):
        self.imu=imu
    # def checkModel(self, pose):
    #     print('pose')
    def count(self):
        self.c += 1


if __name__ == '__main__':
    rospy.init_node('text_listener', anonymous=True)
    rate = rospy.Rate(1)
    i = 0
    odom = odom()
    while not rospy.is_shutdown():
        scan = rospy.wait_for_message('/limo/scan', LaserScan)
        image = rospy.wait_for_message('/limo/color/image_raw', Image)
        pose = rospy.wait_for_message('/gazebo/model_states', ModelStates)
        odom.count()
        print('第', odom.c, '次')
        print(odom.odom.pose.pose.orientation.x, odom.odom.pose.pose.orientation.y,odom.odom.pose.pose.orientation.z)
        # print('orientation',odom.imu.orientation.x,odom.imu.orientation.y)
        # print('linear_acceleration',odom.imu.linear_acceleration.x,odom.imu.linear_acceleration.y)
        # print('angular_velocity',odom.imu.angular_velocity.x,odom.imu.angular_velocity.y)
        rate.sleep()
        # break
