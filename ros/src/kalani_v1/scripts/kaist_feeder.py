#!/usr/bin/env python

import numpy as np
import time
import thread
import sys

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, MagneticField, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

from utilities import *
from kaist_datahandle import *


def timer():
    global pause, step
    log.log('Timer started.')
    prev_time = time.time()
    while True:
        duration = time.time() - prev_time
        if duration > 0.0001:
            global current_time
            current_time += duration * rate
            prev_time += duration
            clock_pub.publish(rospy.Time.from_sec(current_time))
        time.sleep(0.00001)
        while pause:
            prev_time = time.time() - 0.001
            if step:
                step = False
                break
            pass


def publish_data():
    next_data = pl.next()
    global current_time
    current_time = next_data[2]
    thread.start_new_thread(timer, ())
    log.log('Publisher started. Starting time: {} s'.format(current_time))
    while next_data is not None:
        if current_time > next_data[2]:
            name = next_data[0]

            if name == kd.GNSS_CLASS_NAME:
                msg = Odometry()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_gnss_receiver']
                msg.pose.pose.position.x = kd.gnss.x[next_data[1]]
                msg.pose.pose.position.y = kd.gnss.y[next_data[1]]
                msg.pose.covariance = np.diag(np.concatenate([sequence_config['var_gnss_position'], np.zeros(4)])).ravel()
                gnss_pub.publish(msg)

            elif name == kd.ALTITUDE_CLASS_NAME:
                msg = Odometry()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_altimeter']
                msg.pose.pose.position.z = kd.altitude.z[next_data[1]]
                msg.pose.covariance = np.diag(np.concatenate([[0, 0], sequence_config['var_altitude'], [0, 0, 0]])).ravel()
                altitude_pub.publish(msg)

            elif name == kd.IMU_CLASS_NAME:
                msg_imu = Imu()
                msg_imu.header.stamp = rospy.Time.from_sec(next_data[2])
                msg_imu.header.frame_id = general_config['tf_frame_imu']
                msg_imu.linear_acceleration.x = kd.imu.acceleration.x[next_data[1]]
                msg_imu.linear_acceleration.y = kd.imu.acceleration.y[next_data[1]]
                msg_imu.linear_acceleration.z = kd.imu.acceleration.z[next_data[1]]
                msg_imu.linear_acceleration_covariance = np.diag(sequence_config['var_imu_linear_acceleration']).ravel()
                msg_imu.angular_velocity.x = kd.imu.angular_rate.x[next_data[1]]
                msg_imu.angular_velocity.y = kd.imu.angular_rate.y[next_data[1]]
                msg_imu.angular_velocity.z = kd.imu.angular_rate.z[next_data[1]]
                msg_imu.angular_velocity_covariance = np.diag(sequence_config['var_imu_angular_velocity']).ravel()
                imu_pub.publish(msg_imu)

                msg_mag = MagneticField()
                msg_mag.header.stamp = rospy.Time.from_sec(next_data[2])
                msg_mag.header.frame_id = general_config['tf_frame_magnetometer']
                msg_mag.magnetic_field.x = kd.imu.magnetic_field.x[next_data[1]]
                msg_mag.magnetic_field.y = kd.imu.magnetic_field.y[next_data[1]]
                msg_mag.magnetic_field.z = kd.imu.magnetic_field.z[next_data[1]]
                msg_mag.magnetic_field_covariance = np.diag(sequence_config['var_magnetometer_orientation']).ravel()
                magneto_pub.publish(msg_mag)

            elif name == kd.VLP_LEFT_CLASS_NAME:
                msg = PointCloud2()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_lidar']
                pointcloud_pub.publish(kd.vlpLeft.get_point_cloud2(msg.header))

            next_data = pl.next()


if __name__ == '__main__':
    kaist_config = get_config_dict()['kaist_dataset']
    sequence_config = kaist_config[kaist_config['sequence']]
    general_config = get_config_dict()['general']
    log = Log(kaist_config['feeder_node_name'])
    rospy.init_node(kaist_config['feeder_node_name'], anonymous=True)
    log.log('Node initialized.')

    current_time = -1
    rate = 0.5
    pause = True
    step = False

    gnss_pub = rospy.Publisher(general_config['processed_gnss_topic'], Odometry, queue_size=1)
    altitude_pub = rospy.Publisher(general_config['processed_altitude_topic'], Odometry, queue_size=1)
    imu_pub = rospy.Publisher(general_config['processed_imu_topic'], Imu, queue_size=1)
    magneto_pub = rospy.Publisher(general_config['processed_magneto_topic'], MagneticField, queue_size=1)

    # TODO: change pointcloud topic
    pointcloud_pub = rospy.Publisher('/raw_data/velodyne_points', PointCloud2, queue_size=1)

    clock_pub = rospy.Publisher('/clock', Clock, queue_size=1)

    sw = Stopwatch()
    sw.start()
    kd = KAISTData()
    kd.load_data(groundtruth=False)
    log.log('Data loaded. Time elapsed: {} s'.format(sw.stop()))

    # publish lidar's transformation w.r.t the vehicle
    tf2_static_broadcaster = tf2.StaticTransformBroadcaster()
    vehicle2lidar_static_tf = TransformStamped()
    vehicle2lidar_static_tf.header.stamp = rospy.Time.from_sec(kd.imu.time[2000])
    vehicle2lidar_static_tf.header.frame_id = general_config['tf_frame_state']
    vehicle2lidar_static_tf.child_frame_id = general_config['tf_frame_lidar']
    vehicle2lidar_static_tf.transform.translation.x = kd.calibrations.VEHICLE_R_LEFTVLP[0, 3]
    vehicle2lidar_static_tf.transform.translation.y = kd.calibrations.VEHICLE_R_LEFTVLP[1, 3]
    vehicle2lidar_static_tf.transform.translation.z = kd.calibrations.VEHICLE_R_LEFTVLP[2, 3]
    vehicle2lidar_quaternion = tft.quaternion_from_matrix(kd.calibrations.VEHICLE_R_LEFTVLP)
    vehicle2lidar_static_tf.transform.rotation.x = vehicle2lidar_quaternion[0]
    vehicle2lidar_static_tf.transform.rotation.y = vehicle2lidar_quaternion[1]
    vehicle2lidar_static_tf.transform.rotation.z = vehicle2lidar_quaternion[2]
    vehicle2lidar_static_tf.transform.rotation.w = vehicle2lidar_quaternion[3]
    tf2_static_broadcaster.sendTransform(vehicle2lidar_static_tf)

    # set initial values (for unobserved biases and velocity)
    rospy.set_param('/kalani/init/var_imu_linear_acceleration_bias', sequence_config['var_imu_linear_acceleration_bias'])
    rospy.set_param('/kalani/init/var_imu_angular_velocity_bias', sequence_config['var_imu_angular_velocity_bias'])
    rospy.set_param('/kalani/init/init_velocity', sequence_config['init_velocity'])
    rospy.set_param('/kalani/init/init_var_velocity', sequence_config['init_var_velocity'])
    rospy.set_param('/kalani/init/init_imu_linear_acceleration_bias', sequence_config['init_imu_linear_acceleration_bias'])
    rospy.set_param('/kalani/init/init_imu_angular_velocity_bias', sequence_config['init_imu_angular_velocity_bias'])

    # start data player
    pl = kd.get_player(starttime=kd.imu.time[2000])

    try:
        thread.start_new_thread(publish_data, ())
        log.log('Node ready.')
    except Exception as e:
        log.log('Publisher thread error: {}'.format(e.message))

    time.sleep(0.1)
    while True:
        if pause:
            inputStr = "Input command (paused):"
        else:
            inputStr = "Input command:"
        command = raw_input(inputStr)
        command = command.strip()
        command = command.lower()
        if command == "s":
            pause = False
        elif command == "p":
            pause = True
        elif command == "r":
            rateStr = raw_input("Rate value (input 'q' to exit):")
            rateStr = rateStr.strip()
            if not rateStr.isalpha():
                rate = float(rateStr)
                log.log("Rate updated to {}".format(rate))
            else:
                log.log("Rate not updated.")
        elif command == "":
            if pause:
                step = True
        elif command == "q":
            sys.exit(0)
        else:
            log.log("Invalid command.")