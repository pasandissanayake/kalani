#!/usr/bin/env python
import roslib
roslib.load_manifest('beginner_tutorials')

import rospy
from beginner_tutorials.msg import GNSS, IMU, GNSS_N
import tf
import numpy as np
from constants import Constants


R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])


def imu_in_body_frame(data):
    x_imu = 0
    y_imu = 0
    z_imu = 0
    pitch = 0  # x axis
    roll = 0  # y axis
    yaw = 0  # z axis
    axes_type = 'rxyz'  # 'rxyz'-> current frame / 'sxyz'-> fixed frame
    br.sendTransform((x_imu, y_imu, z_imu), tf.transformations.quaternion_from_euler(pitch, roll, yaw, axes_type),
                     rospy.Time.now(), 'imu', 'body')
    # pub_imu = rospy.Publisher(Constants.IMU_DATA_BODY_TOPIC, IMU, queue_size=10)
    # rot = tf.transformations.euler_matrix(pitch, roll, yaw, axes_type)[:3, :3]
    # msg = IMU()
    # msg.header.stamp = data.header.stamp
    # msg.header.frame_id = Constants.BODY_FRAME
    # msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z = np.dot(rot, np.array(
    #     [data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z]))
    # msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = np.dot(rot, np.array(
    #     [data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))
    # msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = np.dot(rot, np.array(
    #     [data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z]))
    # pub_imu.publish(msg)


def gps_in_navigation_frame(data):
    pub_gps = rospy.Publisher(Constants.GNSS_DATA_NAVIGATION_TOPIC, GNSS_N, queue_size=10)
    current = np.array([data.latitude, data.longitude, data.altitude])
    origin = np.array([42.293227 * np.pi / 180, -83.709657 * np.pi / 180, 270])
    a = 6378137
    b = 6356752
    N_c = a ** 2 / np.sqrt((a ** 2) * (np.cos(current[0]) ** 2) + (b ** 2) * (np.sin(current[0]) ** 2))
    N_o = a ** 2 / np.sqrt((a ** 2) * (np.cos(origin[0]) ** 2) + (b ** 2) * (np.sin(origin[0]) ** 2))

    # location of the origin
    x_o = (N_o + origin[2]) * (np.cos(origin[0])) * (np.cos(origin[1]))
    y_o = (N_o + origin[2]) * (np.cos(origin[0])) * (np.sin(origin[1]))
    z_o = (N_o * (1.0 * (b ** 2) / (a ** 2)) + origin[2]) * (np.sin(origin[0]))

    # current location
    x_c = (N_c + current[2]) * (np.cos(current[0])) * (np.cos(current[1]))
    y_c = (N_c + current[2]) * (np.cos(current[0])) * (np.sin(current[1]))
    z_c = (N_c * (1.0 * (b ** 2) / (a ** 2)) + current[2]) * (np.sin(current[0]))

    R_ne = np.array(
        [[-np.cos(origin[1]) * np.sin(origin[0]), -np.sin(origin[0]) * np.sin(origin[1]), np.cos(origin[0])],
         [-np.sin(origin[1]), np.cos(origin[1]), 0],
         [-np.cos(origin[1]) * np.cos(origin[0]), -np.sin(origin[1]) * np.cos(origin[0]), -np.sin(origin[0])]])

    x, y, z = np.dot(R_ne, np.array([x_c - x_o, y_c - y_o, z_c - z_o]))  # GNSS data in NED

    msg = GNSS_N()
    msg.header.stamp = data.header.stamp
    msg.fix_mode = data.fix_mode
    x_new, y_new, z_new = np.dot(R_ned_enu, np.array([x, y, z]))
    msg.x, msg.y, msg.z = [x_new, y_new, z_new]  # GNSS data in ENU
    pub_gps.publish(msg)
    br = tf.TransformBroadcaster()
    br.sendTransform((x_new, y_new, z_new),
                     tf.transformations.quaternion_from_euler(0, 0, 0),
                     rospy.Time.now(),
                     "gnss",
                     "world")


if __name__ == '__main__':
    rospy.init_node('data_broadcaster')
    rospy.Subscriber(Constants.IMU_DATA_TOPIC, IMU, imu_in_body_frame)
    rospy.Subscriber(Constants.GNSS_DATA_TOPIC, GNSS, gps_in_navigation_frame)
    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()
    rospy.spin()
