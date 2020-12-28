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
prev_index_dp = 0

# idx = np.argmin(np.abs(gt.time - gt.time[prev_index] - 1))
# print idx, gt.time[idx] - gt.time[prev_index]
# print gt.x[idx] - gt.x[prev_index]
# print gt.y[idx] - gt.y[prev_index]
# print gt.z[idx] - gt.z[prev_index]


def publish_groundtruth_derived_dp(data):
    global prev_index_dp
    time = data.header.stamp.to_sec()
    new_index_dp = np.argmin(np.abs(gt.time - time))
    # new_index_dp = np.argmin(np.abs(od.time - time))

    n = np.random.multivariate_normal(np.zeros(3), np.eye(3) * 0.01)

    msg = Odometry()
    msg.header.frame_id = 'world'

    # groundtruth derived dp
    msg.header.stamp = rospy.Time.from_sec(gt.time[new_index_dp])
    msg.pose.pose.position.x = gt.x[new_index_dp] - gt.x[prev_index_dp] + n[0]
    msg.pose.pose.position.y = gt.y[new_index_dp] - gt.y[prev_index_dp] + n[1]
    msg.pose.pose.position.z = gt.z[new_index_dp] - gt.z[prev_index_dp] + n[2]

    # actual odometry dp
    # msg.header.stamp = rospy.Time.from_sec(od.time[new_index_dp])
    # msg.pose.pose.position.x = od.x[new_index_dp] - od.x[prev_index_dp]
    # msg.pose.pose.position.y = od.y[new_index_dp] - od.y[prev_index_dp]
    # msg.pose.pose.position.z = od.z[new_index_dp] - od.z[prev_index_dp]

    print "dp: ", msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z

    dp_pub.publish(msg)

    prev_index_dp = new_index_dp


def publish_groundtruth_derived_gnss(data):
    publish_groundtruth_derived_dp(data)
    print 'received gnss'

    time = data.header.stamp.to_sec()
    new_index_gnss = np.argmin(np.abs(gt.time - time))

    n = np.random.multivariate_normal(np.zeros(3), np.eye(3) * 25)

    msg = NavSatFix()
    msg.header.stamp = rospy.Time.from_sec(gt.time[new_index_gnss])
    msg.header.frame_id = 'world'
    msg.status.status = 3
    msg.latitude =  gt.x[new_index_gnss] + n[0]
    msg.longitude = gt.y[new_index_gnss] + n[1]
    msg.altitude =  gt.z[new_index_gnss] + n[2]

    print msg.latitude, msg.longitude

    gnss_mod_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('gt_dp', anonymous=True)

    # rospy.Subscriber('raw_data/gnss_fix', NavSatFix, publish_groundtruth_derived_dp, queue_size=1)
    rospy.Subscriber('raw_data/gnss_fix', NavSatFix, publish_groundtruth_derived_gnss, queue_size=1)

    # publisher for groundtruth derived dp
    dp_pub = rospy.Publisher('/laser_odom_dt', Odometry, queue_size=1)

    # publisher to publish gnss readings derived from ground truth
    gnss_mod_pub = rospy.Publisher('raw_data/gnss_fix_mod', NavSatFix, queue_size=1)

    print 'started...'

    rospy.spin()

