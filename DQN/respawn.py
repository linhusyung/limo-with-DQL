#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
import os
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose

class reset():
    def __init__(self):
        # rospy.init_node('reset',anonymous=True)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.msg = Twist()
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)

    def reset_world(self):
        # 重置世界
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
            rospy.set_param('/done', 0)
        except:
            print("gazebo/gazebo/reset_world service call failed")

    def reset_and_stop(self):
        # 重置世界並且往vel_cmd發佈線速度角速度=0 後台變數done=0
        # rospy.wait_for_service('gazebo/reset_simulation')
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
            self.msg.linear.x = float(0)
            self.msg.angular.z = float(0)
            self.pub.publish(self.msg)
            rospy.set_param('/done', 0)
        except:
            # print("gazebo/reset_simulation service call failed")
            print("gazebo/reset_world service call failed")

    def stop(self):
        self.msg.linear.x = float(0)
        self.msg.angular.z = float(0)
        self.pub.publish(self.msg)

    def delet_model(self):
        # 删除模型
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            remove_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            remove_model_proxy("model_1")
            # remove_model_proxy("some_robo_name")
        except:
            print("Service call delete_model failed: %e")

    def add_finish_model(self, x: str, y: str):
        # 加入終點
        GAZEBO_MODEL_PATH = "/home/a/limo_ws/src/ugv_sim/limo/limo_DRL/DQN/limo_with_DRL/model"
        self.model_to_add = 'model'  # string
        os.system(
            "rosrun gazebo_ros spawn_model -file " + GAZEBO_MODEL_PATH + "/model.sdf -sdf -model " + self.model_to_add + "_1 -x " + x + " -y " + y)

    def chage_finish_model_pose(self):
        pass

    def SpawnModel(self, x: int, y: int):
        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        f = open(
            '/home/a/limo_ws/src/ugv_sim/limo/limo_DRL/DQN/limo_with_DRL_img _scan/model/model.sdf',
            'r')
        sdff = f.read()
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox("model_1", sdff, "robotos_name_space", initial_pose, "world")