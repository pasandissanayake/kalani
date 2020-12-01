#!/usr/bin/env python2

import numpy as np
import numdifftools as numdiff

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from kalman_filter import KalmanFilter
from utilities import *


# placeholders for most recent measurement values
gnss_fix = None
linear_acceleration = None
angular_acceleration = None
magnetic_field = None
altitude = None


def gnss_callback(data):
    global gnss_fix
    t = data.header.stamp.to_sec()
    gnss_fix = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
    cov = np.np.reshape(data.pose.covariance, (3,3))[0:2, 0:2]
    if kf.is_initialized():
        def h(state):
            return state[0:2]
        def meas_func(state):
            return gnss_fix - h(state)
        def hx_func(state):
            jacob_h = numdiff.Jacobian(h)
            return jacob_h(state)
        kf.correct(meas_func, hx_func, cov, t, measurementname='gnss')
    else:
        kf.initialize_state(
            p=gnss_fix, cov_p=np.diagonal(cov),
            v=init_velocity, cov_v=init_var_velocity,
            ab=init_imu_linear_acceleration_bias, cov_ab=var_imu_linear_acceleration_bias,
            wb=init_imu_angular_velocity_bias, cov_wb=var_imu_angular_velocity_bias,
            time=t
        )


def imu_callback(data):
    global linear_acceleration, angular_velocity
    t = data.header.stamp.to_sec()
    linear_acceleration = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
    angular_velocity = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
    var_am = np.diagonal(np.reshape(data.linear_acceleration.covariance, (3,3)))
    var_wm = np.diagonal(np.reshape(data.linear_acceleration.covariance, (3,3)))
    kf.predict(linear_acceleration, var_am, angular_velocity, var_wm, var_imu_linear_acceleration_bias, var_imu_angular_velocity_bias, t)


def mag_callback(data):
    pass


def laserodom_callback(data):
    pass


def publish_static_transforms(static_broadcastor):
    # world to map static transformation
    world2map_static_tf = TransformStamped()
    world2map_static_tf.header.stamp = rospy.Time.now()
    world2map_static_tf.header.frame_id = config['tf_frame_world']
    world2map_static_tf.child_frame_id = config['tf_frame_map']
    world2map_static_tf.transform.translation.x = 0
    world2map_static_tf.transform.translation.y = 0
    world2map_static_tf.transform.translation.z = 0
    world2map_static_tf.transform.rotation.w = 1
    world2map_static_tf.transform.rotation.x = 0
    world2map_static_tf.transform.rotation.y = 0
    world2map_static_tf.transform.rotation.z = 0

    # map to odom static transformation
    map2odom_static_tf = TransformStamped()
    map2odom_static_tf.header.stamp = rospy.Time.now()
    map2odom_static_tf.header.frame_id = config['tf_frame_map']
    map2odom_static_tf.child_frame_id = config['tf_frame_odom']
    map2odom_static_tf.transform.translation.x = 0
    map2odom_static_tf.transform.translation.y = 0
    map2odom_static_tf.transform.translation.z = 0
    map2odom_static_tf.transform.rotation.w = 1
    map2odom_static_tf.transform.rotation.x = 0
    map2odom_static_tf.transform.rotation.y = 0
    map2odom_static_tf.transform.rotation.z = 0

    static_broadcastor.sendTransform([world2map_static_tf, map2odom_static_tf])


if __name__ == '__main__':
    config = get_config_dict()['general']
    log = Log(config['locator_node_name'])

    rospy.init_node(config['locator_node_name'], anonymous=True)
    log.log('Node initialized.')

    rospy.Subscriber(config['processed_gnss_topic'], Odometry, gnss_callback, queue_size=1)
    rospy.Subscriber(config['processed_imu_topic'], Imu, imu_callback, queue_size=1)
    rospy.Subscriber(config['processed_magneto_topic'], MagneticField, mag_callback, queue_size=1)
    rospy.Subscriber(config['processed_laserodom_topic'], Odometry, laserodom_callback, queue_size=1)

    state_pub = rospy.Publisher(config['state_topic'], Odometry, queue_size=1)

    tf2_broadcaster = tf2.TransformBroadcaster()
    tf2_static_broadcaster = tf2.StaticTransformBroadcaster()

    tf2_buffer = tf2.Buffer()
    tf2_listener = tf2.TransformListener(tf2_buffer)

    publish_static_transforms(tf2_static_broadcaster)

    # get initial values for unobserved biases and velocity
    var_imu_linear_acceleration_bias  = np.array(rospy.get_param('/kalani/init/var_imu_linear_acceleration_bias') )
    var_imu_angular_velocity_bias     = np.array(rospy.get_param('/kalani/init/var_imu_angular_velocity_bias')    )
    init_velocity                     = np.array(rospy.get_param('/kalani/init/init_velocity')                    )
    init_var_velocity                 = np.array(rospy.get_param('/kalani/init/init_var_velocity')                )
    init_imu_linear_acceleration_bias = np.array(rospy.get_param('/kalani/init/init_imu_linear_acceleration_bias'))
    init_imu_angular_velocity_bias    = np.array(rospy.get_param('/kalani/init/init_imu_angular_velocity_bias')   )

    kf = KalmanFilter()

    log.log('Node ready.')

    rospy.spin()