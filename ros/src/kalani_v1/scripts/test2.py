#!/usr/bin/env python

import rospy
import tf.transformations as tft
import tf
import tf2_py as tf2
import tf2_ros
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from kalani_v1.msg import State
import pcl_ros
import numpy as np
import scipy
from scipy.optimize import leastsq
from filter.kalman_filter_v1 import Kalman_Filter_V1
from datasetutils.kaist_data_conversions import KAISTDataConversions
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_geometry_msgs
import autograd.numpy as anp
from autograd import jacobian
from scipy.interpolate import interp1d
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, Quaternion
from constants import Constants
from datasetutils.kaist_data_conversions import KAISTDataConversions, KAISTData


def get_transform_matrix_from_ts(ts_msg):
    q = np.array([ts_msg.transform.rotation.x, ts_msg.transform.rotation.y, ts_msg.transform.rotation.z, ts_msg.transform.rotation.w])
    t = np.array([ts_msg.transform.translation.x, ts_msg.transform.translation.y, ts_msg.transform.translation.z])
    R = tft.quaternion_matrix(q)
    R[0:3, 3] = t.T
    return R


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.EVALUATOR_NODE_NAME, anonymous=True)
        print 'Node initialized.'

        ds = KAISTData('/home/entc/kalani/data/kaist/urban17', gnss=False)
        gt = np.array([ds.groundtruth.time, ds.groundtruth.x, ds.groundtruth.y, ds.groundtruth.z]).T

        gt_function = interp1d(gt[:, 0], gt[:, 1:4], axis=0, bounds_error=False, fill_value=gt[0, 1:7], kind='linear')
        print 'Interpolation finished.'

        tf2_buffer = tf2_ros.Buffer()
        tf2_listener = tf2_ros.TransformListener(tf2_buffer)

        print 'Writer ready.'

        rate = rospy.Rate(5)
        k = 0
        while True:
            try:
                if tf2_buffer.can_transform('odom', 'laser_data', rospy.Time(0)):
                    odom_to_laser_data = tf2_buffer.lookup_transform('odom', 'laser_data', rospy.Time(0))
                    R = get_transform_matrix_from_ts(odom_to_laser_data)
                    laser_pos = R[0:3, 3]

                    time = odom_to_laser_data.header.stamp.to_sec()
                    gt_pos = gt_function(time)

                    f_gt = open('groundtruth.csv', 'a+')
                    f_la = open('laser.csv', 'a+')
                    f_la.write(str(time) + ', ' + str(laser_pos[0]) + ', ' + str(laser_pos[1]) + ', ' + str(laser_pos[2]) + '\n')
                    f_gt.write(str(time) + ', ' + str(gt_pos[0]) + ', ' + str(gt_pos[1]) + ', ' + str(gt_pos[2]) + '\n')
                    f_gt.close()
                    f_la.close()

                    k += 1

            except Exception as e:
                print e.message

            rate.sleep()

        print 'Process ended.'

    except rospy.ROSInterruptException:
        print 'Stopping node unexpectedly.'