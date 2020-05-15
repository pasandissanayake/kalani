#!/usr/bin/env python

import rospy
from kalani_v1.msg import IMU
from kalani_v1.msg import State

import numpy as np
import pandas as pd

from constants import Constants


df=pd.read_csv(Constants.IMU_DATA_PATH, header=None)
state_initialized = False


def log(message):
    rospy.loginfo(Constants.IMU_NODE_NAME + ' := ' + str(message))


def state_callback(data):
    global state_initialized
    state_initialized = data.is_initialized


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.IMU_NODE_NAME, anonymous=True)
        rospy.Subscriber(Constants.STATE_TOPIC, State, state_callback)
        pub = rospy.Publisher(Constants.IMU_DATA_TOPIC, IMU, queue_size=10)
        msg = IMU()

        i_imu = 0
        log('Node initialized. Sending data.')
        while not rospy.is_shutdown() and len(df) > i_imu:
            imu_input = list(df.loc[i_imu])

            msg.header.stamp = rospy.Time.from_sec(imu_input[0] * 10 ** (-6))
            msg.header.frame_id = Constants.IMU_FRAME
            msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z = imu_input[1:4] * 10 ** (-4)
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = imu_input[4:7]
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = imu_input[7:10]
            pub.publish(msg)

            if state_initialized: i_imu = i_imu + 1
            if i_imu < len(df) - 1:
                t = df.loc[i_imu + 1][0] - df.loc[i_imu][0]
                rospy.sleep(t * 10 ** (-6))
        log('End of data file. Stopping node gracefully.')

    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
