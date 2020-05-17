#!/usr/bin/env python

import rospy
from kalani_v1.msg import State

import numpy as np
import pandas as pd

from filter.rotations_v1 import Quaternion
from constants import Constants


df=pd.read_csv(Constants.GROUNDTRUTH_DATA_PATH, header=None)


def log(message):
    rospy.loginfo(Constants.NCLT_GROUNDTRUTH_NODE_NAME + ' := ' + str(message))


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.NCLT_GROUNDTRUTH_NODE_NAME, anonymous=True)
        pub = rospy.Publisher(Constants.GROUNDTRUTH_DATA_TOPIC, State, queue_size=10)
        i_gt = 0
        log('Node initialized. Sending data.')
        R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        while not rospy.is_shutdown() and len(df) > i_gt:
            gt_input = np.array(list(df.loc[i_gt]))

            gt_input[1:4] = np.dot(R_ned_enu,gt_input[1:4])
            gt_input[4:7] = np.dot(R_ned_enu,gt_input[4:7])

            msg = State()
            msg.header.stamp = rospy.Time.from_sec(gt_input[0] / 1e6)
            msg.position.x, msg.position.y, msg.position.z = list(gt_input[1:4])
            msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = Quaternion(euler=gt_input[4:7]).to_numpy().tolist()
            pub.publish(msg)
            pub.publish(msg)

            if i_gt < len(df) - 1:
                dt = (df.loc[i_gt + 1][0] - df.loc[i_gt][0]) * 10 ** (-6)
                rospy.sleep(dt)
            i_gt += 1
        log('End of data file. Stopping node gracefully.')

    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
