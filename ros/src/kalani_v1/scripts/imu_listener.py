#!/usr/bin/env python


import rospy
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard imu data %s', data.data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('data_listener_imu', anonymous=True)
    rospy.Subscriber('imu_data', numpy_msg(Floats),callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
