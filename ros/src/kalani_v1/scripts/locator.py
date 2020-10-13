#!/usr/bin/env python2

import numpy as np

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from kalmanfilter import KalmanFilter
from utilities import *


def gnss_callback(data):
    pass


def imu_callback(data):
    pass


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

    kf = KalmanFilter()

    log.log('Node ready.')

    rospy.spin()