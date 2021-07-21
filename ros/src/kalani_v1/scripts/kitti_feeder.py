#!/usr/bin/python2

import numpy as np
import time
import thread
import sys

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, MagneticField, PointCloud2, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import (Float32MultiArray, Float64MultiArray,
                          Int8MultiArray, Int16MultiArray,
                          Int32MultiArray, Int64MultiArray,
                          UInt8MultiArray, UInt16MultiArray,
                          UInt32MultiArray, UInt64MultiArray)
from cv_bridge import CvBridge

from utilities import *
from kitti_datahandle import *
import ros_np_multiarray as rnm


def timer():
    global pause, step
    log.log('Timer started.')
    prev_time = time.time()
    while True:
        duration = time.time() - prev_time
        if duration > 0.001:
            global current_time
            current_time += duration * rate
            prev_time += duration
            clock_pub.publish(rospy.Time.from_sec(current_time))
            log.log('time: ', current_time)
        time.sleep(0.001)
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
            # log.log('current time:{}, timestamp:{}, name:{}'.format(current_time, next_data[2],next_data[0]))
            name = next_data[0]

            if name == kd.GNSS_CLASS_NAME:
                msg = Odometry()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_gnss_receiver']
                msg.pose.pose.position.x = kd.gnss.x[next_data[1]]
                msg.pose.pose.position.y = kd.gnss.y[next_data[1]]
                msg.pose.covariance = np.diag(np.concatenate([sensor_config['var_gnss_position'], np.zeros(4)])).ravel()
                gnss_pub.publish(msg)

            elif name == kd.ALTITUDE_CLASS_NAME:
                msg = Odometry()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_altimeter']
                msg.pose.pose.position.z = kd.altitude.z[next_data[1]]
                msg.pose.covariance = np.diag(np.concatenate([[0, 0], sensor_config['var_altitude'], [0, 0, 0]])).ravel()
                altitude_pub.publish(msg)

            elif name == kd.IMU_CLASS_NAME:
                msg_imu = Imu()
                msg_imu.header.stamp = rospy.Time.from_sec(next_data[2])
                msg_imu.header.frame_id = general_config['tf_frame_imu']
                msg_imu.linear_acceleration.x = kd.imu.acceleration.x[next_data[1]]
                msg_imu.linear_acceleration.y = kd.imu.acceleration.y[next_data[1]]
                msg_imu.linear_acceleration.z = kd.imu.acceleration.z[next_data[1]]
                msg_imu.linear_acceleration_covariance = np.diag(sensor_config['var_imu_linear_acceleration']).ravel()
                msg_imu.angular_velocity.x = kd.imu.angular_rate.x[next_data[1]]
                msg_imu.angular_velocity.y = kd.imu.angular_rate.y[next_data[1]]
                msg_imu.angular_velocity.z = kd.imu.angular_rate.z[next_data[1]]
                msg_imu.angular_velocity_covariance = np.diag(sensor_config['var_imu_angular_velocity']).ravel()
                imu_pub.publish(msg_imu)

            elif name == kd.VLP_CLASS_NAME:
                msg = PointCloud2()
                msg.header.stamp = rospy.Time.from_sec(next_data[2])
                msg.header.frame_id = general_config['tf_frame_lidar']
                msg.is_dense = True
                pointcloud , tracklet = kd.vlp.get_point_cloud2(msg.header)
                pointcloud_pub.publish(pointcloud)
                tracklet_pub.publish(rnm.to_multiarray_f32(np.array( tracklet , dtype=np.float32)))

            elif name == kd.STEREO_IMAGE_CLASS_NAME:
                limage, rimage, cimage, tracklet = kd.stereoImage.get_stereo_images(next_data[2])
                height = np.shape(limage)[0]
                width = np.shape(limage)[1]
                lmsg = Image()
                lmsg.header.stamp = rospy.Time.from_sec(next_data[2])
                lmsg.header.frame_id = general_config['tf_frame_camera']
                lmsg.height = height
                lmsg.width = width
                lmsg.step = width
                lmsg.encoding = 'mono8'
                lmsg.data = limage.flatten().tolist()
                rmsg = Image()
                rmsg.header.stamp = rospy.Time.from_sec(next_data[2])
                rmsg.header.frame_id = general_config['tf_frame_camera']
                rmsg.height = height
                rmsg.width = width
                rmsg.step = width
                rmsg.encoding = 'mono8'
                rmsg.data = rimage.flatten().tolist()
                stereo_left_pub.publish(lmsg)
                stereo_right_pub.publish(rmsg)
                stereo_colour_pub.publish(bridge.cv2_to_imgmsg(cimage, "bgr8"))
                #tracklet_pub.publish(rnm.to_multiarray_f32(np.array( tracklet , dtype=np.float32)))
                
            next_data = pl.next()
            time.sleep(0.001)
            # log.log('current time:{}, timestamp:{}, name:{}, time:{}'.format(current_time, next_data[2],next_data[0],sw.stop()))


if __name__ == '__main__':
    kitti_config = get_config_dict()['kitti_dataset']
    sensor_config = kitti_config['sensor_characteristics']
    date = kitti_config['date']
    drive = kitti_config['drive']
    sequence = 's{}_{}'.format(date, drive)
    sequence_config = kitti_config[sequence]
    general_config = get_config_dict()['general']
    log = Log(kitti_config['feeder_node_name'])
    rospy.init_node(kitti_config['feeder_node_name'], anonymous=True)
    log.log('Node initialized.')

    current_time = -1
    rate = 0.5
    pause = True
    step = False

    gnss_pub = rospy.Publisher(general_config['processed_gnss_topic'], Odometry, queue_size=1)
    altitude_pub = rospy.Publisher(general_config['processed_altitude_topic'], Odometry, queue_size=1)
    imu_pub = rospy.Publisher(general_config['processed_imu_topic'], Imu, queue_size=1)
    stereo_left_pub = rospy.Publisher(general_config['raw_stereo_image_left'], Image, queue_size=1)
    stereo_right_pub = rospy.Publisher(general_config['raw_stereo_image_right'], Image, queue_size=1)
    stereo_colour_pub = rospy.Publisher(general_config['raw_colour_image'], Image, queue_size=1)
    pointcloud_pub = rospy.Publisher(general_config['raw_pointcloud'], PointCloud2, queue_size=1)
    tracklet_pub = rospy.Publisher('tracklets', Float32MultiArray , queue_size=0)

    clock_pub = rospy.Publisher('/clock', Clock, queue_size=1)

    bridge = CvBridge()

    sw = Stopwatch()
    sw.start()
    kd = KITTIData()
    kd.load_data(oxts=True, vlp=True, stereoimage=True, calibrations=True)
    log.log('Data loaded. Time elapsed: {} s'.format(sw.stop()))

    tf2_static_broadcaster = tf2.StaticTransformBroadcaster()
    # lidar's transformation w.r.t the vehicle
    vehicle2lidar_static_tf = TransformStamped()
    vehicle2lidar_static_tf.header.stamp = rospy.Time.from_sec(kd.imu.time[0])
    vehicle2lidar_static_tf.header.frame_id = general_config['tf_frame_state']
    vehicle2lidar_static_tf.child_frame_id = general_config['tf_frame_lidar']
    vehicle2lidar_static_tf.transform.translation.x = kd.calibrations.VEHICLE_R_VLP[0, 3]
    vehicle2lidar_static_tf.transform.translation.y = kd.calibrations.VEHICLE_R_VLP[1, 3]
    vehicle2lidar_static_tf.transform.translation.z = kd.calibrations.VEHICLE_R_VLP[2, 3]
    vehicle2lidar_quaternion = tft.quaternion_from_matrix(kd.calibrations.VEHICLE_R_VLP)
    vehicle2lidar_static_tf.transform.rotation.x = vehicle2lidar_quaternion[0]
    vehicle2lidar_static_tf.transform.rotation.y = vehicle2lidar_quaternion[1]
    vehicle2lidar_static_tf.transform.rotation.z = vehicle2lidar_quaternion[2]
    vehicle2lidar_static_tf.transform.rotation.w = vehicle2lidar_quaternion[3]

    # stereo camera's transformation w.r.t. the vehicle
    vehicle2stereo_static_tf = TransformStamped()
    vehicle2stereo_static_tf.header.stamp = rospy.Time.from_sec(kd.imu.time[0])
    vehicle2stereo_static_tf.header.frame_id = general_config['tf_frame_state']
    vehicle2stereo_static_tf.child_frame_id = general_config['tf_frame_camera']
    vehicle2stereo_static_tf.transform.translation.x = kd.calibrations.VEHICLE_R_STEREO[0, 3]
    vehicle2stereo_static_tf.transform.translation.y = kd.calibrations.VEHICLE_R_STEREO[1, 3]
    vehicle2stereo_static_tf.transform.translation.z = kd.calibrations.VEHICLE_R_STEREO[2, 3]
    vehicle2stereo_quaternion = tft.quaternion_from_matrix(kd.calibrations.VEHICLE_R_STEREO)
    vehicle2stereo_static_tf.transform.rotation.x = vehicle2stereo_quaternion[0]
    vehicle2stereo_static_tf.transform.rotation.y = vehicle2stereo_quaternion[1]
    vehicle2stereo_static_tf.transform.rotation.z = vehicle2stereo_quaternion[2]
    vehicle2stereo_static_tf.transform.rotation.w = vehicle2stereo_quaternion[3]

    tf2_static_broadcaster.sendTransform([vehicle2lidar_static_tf, vehicle2stereo_static_tf])

    # set initial values (for unobserved biases and velocity)
    rospy.set_param('/kalani/init/var_imu_linear_acceleration_bias', sensor_config['var_imu_linear_acceleration_bias'])
    rospy.set_param('/kalani/init/var_imu_angular_velocity_bias', sensor_config['var_imu_angular_velocity_bias'])
    rospy.set_param('/kalani/init/init_velocity', sensor_config['init_velocity'])
    rospy.set_param('/kalani/init/init_var_velocity', sensor_config['init_var_velocity'])
    rospy.set_param('/kalani/init/init_imu_linear_acceleration_bias', sensor_config['init_imu_linear_acceleration_bias'])
    rospy.set_param('/kalani/init/init_imu_angular_velocity_bias', sensor_config['init_imu_angular_velocity_bias'])

    # start data player
    starttime = timestamp_from_kitti_string(str(sequence_config['time_start_motion']))
    pl = kd.get_player(starttime=starttime)

    try:
        thread.start_new_thread(publish_data, ())
        log.log('Node ready.')
        log.log('Start time: {}, rate: {}'.format(starttime, rate))
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