#!/usr/bin/env python

import numpy as np
import time
import thread

import rospy
import tf.transformations as tft
import tf2_ros

from sensor_msgs.msg import Imu, MagneticField, PointCloud2
from nav_msgs.msg import Odometry

from utilities import *
from kaist_datahandle import *


def timer():
    log.log('Timer started.')
    prev_time = time.time()
    while True:
        duration = time.time() - prev_time
        if duration > 0.0001:
            global current_time
            current_time += duration
            prev_time += duration
        time.sleep(0.00001)


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
                gnss_pub.publish(msg)

            elif name == kd.ALTITUDE_CLASS_NAME:
                msg = Odometry()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_altimeter']
                msg.pose.pose.position.z = kd.altitude.z[next_data[1]]
                altitude_pub.publish(msg)

            elif name == kd.IMU_CLASS_NAME:
                msg_imu = Imu()
                msg_imu.header.stamp = rospy.Time.from_sec(next_data[2])
                msg_imu.header.frame_id = general_config['tf_frame_imu']
                msg_imu.linear_acceleration.x = kd.imu.acceleration.x[next_data[1]]
                msg_imu.linear_acceleration.y = kd.imu.acceleration.y[next_data[1]]
                msg_imu.linear_acceleration.z = kd.imu.acceleration.z[next_data[1]]
                msg_imu.angular_velocity.x = kd.imu.angular_rate.x[next_data[1]]
                msg_imu.angular_velocity.y = kd.imu.angular_rate.y[next_data[1]]
                msg_imu.angular_velocity.z = kd.imu.angular_rate.z[next_data[1]]
                imu_pub.publish(msg_imu)

                msg_mag = MagneticField()
                msg_mag.header.stamp = rospy.Time.from_sec(next_data[2])
                msg_mag.header.frame_id = general_config['tf_frame_magnetometer']
                msg_mag.magnetic_field.x = kd.imu.magnetic_field.x[next_data[1]]
                msg_mag.magnetic_field.y = kd.imu.magnetic_field.y[next_data[1]]
                msg_mag.magnetic_field.z = kd.imu.magnetic_field.z[next_data[1]]
                magneto_pub.publish(msg_mag)

            next_data = pl.next()


if __name__ == '__main__':
    kaist_config = get_config_dict()['kaist_dataset']
    general_config = get_config_dict()['general']
    log = Log(kaist_config['feeder_node_name'])
    rospy.init_node(kaist_config['feeder_node_name'], anonymous=True)
    log.log('Node initialized.')

    current_time = -1

    gnss_pub = rospy.Publisher(general_config['processed_gnss_topic'], Odometry, queue_size=1)
    altitude_pub = rospy.Publisher(general_config['processed_altitude_topic'], Odometry, queue_size=1)
    imu_pub = rospy.Publisher(general_config['processed_imu_topic'], Imu, queue_size=1)
    magneto_pub = rospy.Publisher(general_config['processed_magneto_topic'], MagneticField, queue_size=1)
    pointcloud_pub = rospy.Publisher('/raw_data/velodyne_points', PointCloud2, queue_size=1)

    sw = Stopwatch()
    sw.start()
    kd = KAISTData()
    kd.load_data(groundtruth=False)
    log.log('Data loaded. Time elapsed: {} s'.format(sw.stop()))

    pl = kd.get_player(starttime=kd.imu.time[2000])
    try:
        thread.start_new_thread(publish_data, ())
        log.log('Node ready.')
    except Exception as e:
        log.log('Publisher thread error: {}'.format(e.message))

    rospy.spin()