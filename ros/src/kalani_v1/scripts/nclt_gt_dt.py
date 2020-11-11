#!/usr/bin/env python

import rospy
from datasetutils.kaist_data_conversions import *
from datasetutils.nclt_data_conversions import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import tf.transformations as tft
import numdifftools as nd
from filter.rotations_v1 import rpy_jacobian_axis_angle
from constants import Constants
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu

ds = NCLTData(Constants.NCLT_DATASET_DIRECTORY)
gt = ds.groundtruth
od = ds.odom
prev_index = 0

# idx = np.argmin(np.abs(gt.time - gt.time[prev_index] - 1))
# print idx, gt.time[idx] - gt.time[prev_index]
# print gt.x[idx] - gt.x[prev_index]
# print gt.y[idx] - gt.y[prev_index]
# print gt.z[idx] - gt.z[prev_index]


def the_callback(data):
    global prev_index

    print 'received gnss'

    time = data.header.stamp.to_sec()
    # new_index = np.argmin(np.abs(gt.time - time))
    new_index = np.argmin(np.abs(od.time - time))

    n = np.random.multivariate_normal(np.zeros(3), np.eye(3) * 0.01)

    msg = Odometry()
    msg.header.frame_id = 'world'

    # msg.header.stamp = rospy.Time.from_sec(gt.time[new_index])
    # msg.pose.pose.position.x = gt.x[new_index] - gt.x[prev_index] + n[0]
    # msg.pose.pose.position.y = gt.y[new_index] - gt.y[prev_index] + n[1]
    # msg.pose.pose.position.z = gt.z[new_index] - gt.z[prev_index] + n[2]

    msg.header.stamp = rospy.Time.from_sec(od.time[new_index])
    msg.pose.pose.position.x = od.x[new_index] - od.x[prev_index] # + n[0]
    msg.pose.pose.position.y = od.y[new_index] - od.y[prev_index] # + n[1]
    msg.pose.pose.position.z = od.z[new_index] - od.z[prev_index] # + n[2]

    dp_pub.publish(msg)

    prev_index = new_index

if __name__ == '__main__':
    rospy.init_node('gt_dp', anonymous=True)

    rospy.Subscriber('raw_data/gnss_fix', NavSatFix, the_callback, queue_size=1)

    dp_pub = rospy.Publisher('/laser_odom_dt', Odometry, queue_size=1)

    print 'started...'

    rospy.spin()

