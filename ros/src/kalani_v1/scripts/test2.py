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

start = time.time()
kd = KAISTData()
kd.load_data()
print 'time elapsed: {} s'.format(time.time() - start)

error = kd.groundtruth.interp_z(kd.altitude.time) - kd.altitude.z
mean = np.average(error)
sdev = np.average((error-mean)**2)
print 'mean = {}, standard deviation = {}'.format(mean, sdev)
plt.hist(error, 100)
plt.grid()
plt.legend()
plt.show()
