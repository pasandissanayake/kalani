#!/usr/bin/env python

import rospy
from kalani_v1.msg import GNSS
from kalani_v1.msg import IMU
from kalani_v1.msg import State

import pandas as pd

from constants import Constants


df=pd.read_csv(Constants.GNSS_DATA_PATH, header=None)

state_initialized = False
imu_time = 0.0


def log(message):
    rospy.loginfo(Constants.GNSS_NODE_NAME + ' := ' + str(message))


def state_callback(data):
    global state_initialized
    state_initialized = data.is_initialized


def imu_callback(data):
    global imu_time
    imu_time = data.header.stamp.to_sec()


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.GNSS_NODE_NAME, anonymous=True)
        rospy.Subscriber(Constants.STATE_TOPIC, State, state_callback)
        rospy.Subscriber(Constants.IMU_DATA_TOPIC, IMU, imu_callback)
        pub = rospy.Publisher(Constants.GNSS_DATA_TOPIC, GNSS, queue_size=10)
        i_gps = 0
        log('Node initialized. Sending data.')
        while not rospy.is_shutdown() and len(df) > i_gps:
            gps_input = list(df.loc[i_gps])

            msg = GNSS()
            t = gps_input[0] * 10 ** (-6)
            msg.header.stamp = rospy.Time.from_sec(t)
            msg.header.frame_id = Constants.GNSS_FRAME
            msg.latitude = gps_input[3]
            msg.longitude = gps_input[4]
            msg.altitude = gps_input[5]
            msg.fix_mode = gps_input[1]
            pub.publish(msg)

            if state_initialized and imu_time>=t:
                i_gps = i_gps + 1
            if i_gps < len(df) - 1:
                dt = (df.loc[i_gps + 1][0] - df.loc[i_gps][0]) * 10 ** (-6)
                rospy.sleep(dt)
        log('End of data file. Stopping node gracefully.')

    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
