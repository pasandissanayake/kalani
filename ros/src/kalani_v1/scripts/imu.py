#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import numpy as np
import pandas as pd

from constants import Constants


df=pd.read_csv(Constants.IMU_DATA_PATH, header=None)


def log(message):
    rospy.loginfo(Constants.IMU_NODE_NAME + ' := ' + str(message))


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.IMU_NODE_NAME, anonymous=True)
        pub = rospy.Publisher(Constants.IMU_DATA_TOPIC, numpy_msg(Floats), queue_size=10)
        i_imu = 0
        log('Node initialized. Sending data.')
        while not rospy.is_shutdown() and len(df) > i_imu:
            imu_input = list(df.loc[i_imu])
            imu_input[0] = imu_input[0] * 10 ** (-6)
            imu_input = np.array(imu_input, dtype=np.float32)
            pub.publish(imu_input)
            # log(imu_input)
            # print ('time_imu %s' % rospy.get_time())
            i_imu = i_imu + 1
            if i_imu < len(df) - 1:
                t = df.loc[i_imu + 1][0] - df.loc[i_imu][0]
                rospy.sleep(t * 10 ** (-6))
        log('End of data file. Stopping node gracefully.')

    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
