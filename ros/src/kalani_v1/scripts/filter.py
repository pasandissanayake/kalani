#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import numpy as np

from constants import Constants
from filter.filter_v1 import Filter_V1
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


kf = Filter_V1()
pub = rospy.Publisher(Constants.STATE_TOPIC, numpy_msg(Floats), queue_size=10)

# Rotation matrix from NED to ENU
R_ned_enu = np.array([[0,1,0],[1,0,0],[0,0,-1]])


def log(message):
    rospy.loginfo(Constants.FILTER_NODE_NAME + ' := ' + str(message))


def write_state_to_file():
    state = kf.get_state_as_numpy()
    log('state: ' + str(state))
    # np.savetxt('prediction.csv',state,delimiter=',')


def publish_state():
    state = kf.get_state_as_numpy()
    pub.publish(state)


def gnss_callback(data):
    # log('received gnss: ' + str(data.data))

    # time, fix(2=w/o alt, 3=with alt), lat, long, alt
    gnss = np.array(np.concatenate((data.data[0:2],data.data[3:6]))).flatten()
    origin = np.array([42.293227 * np.pi / 180, -83.709657 * np.pi / 180, 270])
    dif = gnss[2:5] - origin

    # GNSS data in NED
    r = 6400000
    gnss[2] = r * np.sin(dif[0])
    gnss[3] = r * np.cos(origin[0]) * np.sin(dif[1])
    gnss[4] = dif[2]

    # GNSS data in ENU
    gnss[2:5] = (R_ned_enu.dot(gnss[2:5].T)).T

    if kf.state_initialized:
        if gnss[1] == 3:
            kf.correct(gnss[2:5],gnss[0],Filter_V1.GNSS_WITH_ALT)
        elif gnss[1] == 2:
            kf.correct(gnss[2:4],gnss[0],Filter_V1.GNSS_NO_ALT)
        publish_state()
    else:
        if gnss[1] == 3:
            p = gnss[2:5]
            v = np.zeros(3)
            q = Quaternion().to_numpy()
            g = np.array([0,0,-9.8])
            ab = np.zeros(3)
            wb = np.zeros(3)

            P = np.eye(15)

            t = gnss[0]

            kf.initialize_state(p,v,q,g,ab,wb,P,t)

            log('Filter initialized.')


def imu_callback(data):
    # log('received imu: ' + str(data.data))
    if kf.state_initialized:
        am = np.array(data.data[4:7])
        wm = np.array(data.data[7:10])
        t = data.data[0]

        kf.predict(am,wm,t)

        log('state: ' + str(kf.get_state_as_numpy()))
        publish_state()


if __name__ == '__main__':
    rospy.init_node(Constants.FILTER_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber(Constants.GNSS_DATA_TOPIC, numpy_msg(Floats), gnss_callback)
    rospy.Subscriber(Constants.IMU_DATA_TOPIC, numpy_msg(Floats), imu_callback)
    rospy.spin()