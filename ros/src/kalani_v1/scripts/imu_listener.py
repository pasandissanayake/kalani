#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from constants import Constants


def log(message):
    rospy.loginfo(Constants.IMU_LISTENER_NODE_NAME + ' := ' + str(message))


def callback(data):
    log('received imu: ' + str(data.data))


if __name__ == '__main__':
    rospy.init_node(Constants.IMU_LISTENER_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber(Constants.IMU_DATA_TOPIC, numpy_msg(Floats), callback)
    rospy.spin()