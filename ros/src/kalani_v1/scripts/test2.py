#!/usr/bin/env python

import rospy
import tf.transformations as tft
import tf
import numpy as np
from scipy.interpolate import interp1d
# from datasetutils.kaist_data_conversions import KAISTDataConversions, KAISTData
import matplotlib.pyplot as plt
import time
from utilities import *
from kaist_datahandle import KAISTData

import pcl
from pcl import pcl_visualization
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
import thread

# start = time.time()
# ds = KAISTData('/home/pasan/kalani-data/kaist/urban17', gt=True, gnss=True)
# gt = ds.groundtruth
# x_speed = np.diff(gt.x)
# y_speed = np.diff(gt.y)
# z_speed = np.diff(gt.z)
# plt.plot(gt.time[:-1], x_speed, label='x')
# plt.plot(gt.time[:-1], y_speed, label='y')
# plt.plot(gt.time[:-1], z_speed, label='z')
# plt.legend()
# plt.grid()
# plt.show()


# error = kd.groundtruth.interp_z(kd.altitude.time) - kd.altitude.z
# mean = np.average(error)
# sdev = np.average((error-mean)**2)
# print 'mean = {}, standard deviation = {}'.format(mean, sdev)
# plt.hist(error, 100)
# plt.grid()
# plt.legend()
# plt.show()

# a name: 1524212006000733000.bin
#   path: '/home/pasan/kalani-data/kaist/urban17/data/VLP_left/1524212006000733000.bin'

# import struct
# import os
#
# # g = np.loadtxt('/home/pasan/kalani-data/kaist/urban17/data/VLP_left_stamp.csv', delimiter='\n').astype(np.int)
# g = os.listdir('/home/pasan/kalani-data/kaist/urban17/data/VLP_left')
# points = []
# missing_files = 0
# for file in g:
#     filename = '/home/pasan/kalani-data/kaist/urban17/data/VLP_left/{}'.format(file)
#     print filename
#     if os.path.isdir(filename):
#         continue
#     if os.path.isfile(filename):
#         f = open(filename, "rb")
#         s = f.read()
#         count = len(s) / 4
#         w = np.concatenate([[int(file[:-4]) / 1e9, count/4], struct.unpack('f'*count, s)])
#         points.append(w)
#     else:
#         missing_files += 1
# print '# total: {}, # missing: {}, # found: {}'.format(len(g), missing_files, len(g)-missing_files)
# data = points[0][2:]
# data = np.reshape(data, (-1, 4))[:, :3]
# data = data.astype(np.float32)
# print '# '

current_time = -1

def timer():
    print 'timer started.'
    prev_time = time.time()
    while True:
        duration = time.time() - prev_time
        if duration > 0.0001:
            global current_time
            current_time += duration
            prev_time += duration
        time.sleep(0.00001)

def publish_data():
    print 'publisher started.'
    next = pl.next()
    global current_time
    current_time = next[2]
    thread.start_new_thread(timer, ())
    while next is not None:
        if current_time > next[2]:
            name = next[0]
            if name == kd.GNSS_CLASS_NAME:
                msg = Odometry()
                msg.header.stamp = rospy.Time.from_sec(next[2])
                msg.header.frame_id = '/gnss'
                msg.pose.pose.position.x = kd.gnss.x[next[1]]
                msg.pose.pose.position.y = kd.gnss.y[next[1]]
                pub_gnss.publish(msg)
            next = pl.next()

if __name__ == '__main__':
    rospy.init_node('applebee', anonymous=True)
    print 'node initialized.'

    pub = rospy.Publisher('/raw_data/velodyne_points', PointCloud2, queue_size=1)
    pub_gnss = rospy.Publisher('/gnss', Odometry, queue_size=1)

    loading_start = time.time()
    kd = KAISTData()
    kd.load_data(groundtruth=False)
    print 'time elapsed: {} s'.format(time.time() - loading_start)
    pl = kd.get_player()
    try:
        thread.start_new_thread(publish_data, ())
    except Exception as e:
        print 'thread error: {}'.format(e.message)

    rospy.spin()

    # p1 = pcl.PointCloud_PointXYZI(p1)
    # v = pcl_visualization.CloudViewing()
    # v.ShowGrayCloud(p1)
    # a = raw_input()