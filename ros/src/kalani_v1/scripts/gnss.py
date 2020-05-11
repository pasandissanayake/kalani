#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import numpy as np
import pandas as pd
import time

from constants import Constants


df=pd.read_csv(Constants.GNSS_DATA_PATH, header=None)


def log(message):
    rospy.loginfo(Constants.GNSS_NODE_NAME + ' := ' + str(message))


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.GNSS_NODE_NAME, anonymous=True)
        pub = rospy.Publisher(Constants.GNSS_DATA_TOPIC, numpy_msg(Floats), queue_size=10)
        i_gps = 0
        log('Node initialized. Sending data.')
        while not rospy.is_shutdown() and len(df) > i_gps:
            gps_input = list(df.loc[i_gps])
            gps_input[0] = gps_input[0] * 10 ** (-6)
            gps_input = np.array(gps_input, dtype=np.float32)
            pub.publish(gps_input)
            # log(gps_input)
            # print ('time_gps %s' % rospy.get_time())
            i_gps = i_gps + 1
            if i_gps < len(df) - 1:
                t = df.loc[i_gps + 1][0] - df.loc[i_gps][0]
                time.sleep(t * 10 ** (-6))
        log('End of data file. Stopping node gracefully.')

    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
