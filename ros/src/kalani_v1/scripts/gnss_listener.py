#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from constants import Constants


def log(message):
    rospy.loginfo(Constants.GNSS_LISTENER_NODE_NAME + ' := ' + str(message))


def callback(data):
    log('received gnss: %s' + str(data.data))


if __name__ == '__main__':
    rospy.init_node(Constants.GNSS_LISTENER_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber(Constants.GNSS_DATA_TOPIC, numpy_msg(Floats), callback)
    rospy.spin()