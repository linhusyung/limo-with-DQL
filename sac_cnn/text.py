import rospy
from gazebo_msgs.srv import DeleteModel
def delet_model():
    # 删除模型
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        remove_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        remove_model_proxy("model_1")
        # remove_model_proxy("some_robo_name")
    except:
        print("Service call delete_model failed: %e")
if __name__ == '__main__':
    rospy.init_node('t_listener', anonymous=True)
    delet_model()