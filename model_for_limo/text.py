# 測試
import rospy
from sensor_msgs.msg import LaserScan, Image


class text():
    def __init__(self):
        rospy.init_node('text_listener', anonymous=True)

    def sub(self):
        rospy.Subscriber('/limo/scan', LaserScan, self.callback)

    def callback(self, data):
        print(min(data.ranges))


if __name__ == '__main__':
    t = text()
    t.sub()
    rospy.spin()
