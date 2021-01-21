#!/usr/bin/env python2

import numpy as np
import numdifftools as numdiff

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from kalani_v1.msg import State
from kalani_v1.msg import Error
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker
import visualization_msgs

from kaist_datahandle import *
from utilities import *

gt_path = Path()       # stores ground truth path (all historical values)
state_path = Path()    # stores state path (all historical values)


def publish_covariance(time, sigma):
    marker = Marker()
    marker.header.frame_id = general_config['tf_frame_state']
    marker.header.stamp = rospy.Time.from_sec(time)
    marker.ns = 'evaluate/covariance'
    marker.type = visualization_msgs.msg.Marker.SPHERE
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = [0, 0, 0]
    marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = [0, 0, 0, 1]

    marker.scale.x = sigma[0] * 3
    marker.scale.y = sigma[1] * 3
    marker.scale.z = sigma[2] * 3
    marker.color.a = 0.5
    marker.color.r = 100.0
    marker.color.g = 100.0
    marker.color.b = 0.0
    cov_ellipsoid_publisher.publish(marker)


def publish_path(path, publisher, timestamp, frame_id, trans, rot):
    path.header.stamp = rospy.Time.from_sec(timestamp)
    path.header.frame_id = frame_id
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.from_sec(timestamp)
    pose.header.frame_id = frame_id
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = list(trans)
    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = list(rot)
    path.poses.append(pose)
    publisher.publish(path)


def callback(data):
    timestamp = data.header.stamp.to_sec()

    # obtain state transform
    try:
        state_tf = tf_buffer.lookup_transform(general_config['tf_frame_world'], general_config['tf_frame_state'], rospy.Time(0))
        trans = np.array([state_tf.transform.translation.x, state_tf.transform.translation.y, state_tf.transform.translation.z])
        rot = np.array([state_tf.transform.rotation.x, state_tf.transform.rotation.y, state_tf.transform.rotation.z, state_tf.transform.rotation.w])
    except Exception as e:
        log.log('State transform lookup error. {}'.format(e.message))
        return
    rot_euler = np.array(tft.euler_from_quaternion(rot))

    # reshape covariance to obtain only the covariance of translation and rotation
    cov_mat = np.array(data.covariance)
    cov_mat_width = int(np.sqrt(len(cov_mat)))
    cov_mat = cov_mat.reshape((cov_mat_width, cov_mat_width))
    trans_cov = cov_mat[0:3, 0:3]
    rot_cov = cov_mat[6:9, 6:9]

    # ground truth values
    gt_trans = np.array([ds.groundtruth.interp_x(timestamp), ds.groundtruth.interp_y(timestamp), ds.groundtruth.interp_z(timestamp)])
    gt_rot_euler = np.array([ds.groundtruth.interp_r(timestamp), ds.groundtruth.interp_p(timestamp), ds.groundtruth.interp_h(timestamp)])
    gt_rot = np.array(tft.quaternion_from_euler(gt_rot_euler[0], gt_rot_euler[1], gt_rot_euler[2]))

    # error values
    error_trans = trans - gt_trans
    error_trans_abs = np.abs(error_trans)
    error_trans_euc = tft.vector_norm(error_trans)
    error_rot = np.array(tft.quaternion_multiply(tft.quaternion_conjugate(gt_rot), rot))
    error_rot_euler = np.array(tft.euler_from_quaternion(error_rot))
    error_rot_euler_abs = np.abs(error_rot_euler)

    # first-order approximation of RPY covariance
    angle, axis = quaternion_to_angle_axis(error_rot)
    J = rpy_jacobian_axis_angle(angle * axis)
    rot_cov_euler = np.matmul(J, np.matmul(rot_cov, J.T))

    # obtain std dev values from estimated the covariance matrices
    trans_sigma = np.sqrt(np.array([trans_cov[0,0], trans_cov[1,1], trans_cov[2,2]]))
    rot_sigma = np.sqrt(np.array([rot_cov_euler[0,0], rot_cov_euler[1,1], rot_cov_euler[2,2]]))

    # obtain gnss, magnetometer transforms from tf
    try:
        gnss_tf = tf_buffer.lookup_transform(general_config['tf_frame_world'], general_config['tf_frame_gnss'], rospy.Time(0))
        gnss_trans = np.array([gnss_tf.transform.translation.x, gnss_tf.transform.translation.y, gnss_tf.transform.translation.z])
    except Exception as e:
        log.log('GNSS transform lookup error. {}'.format(e.message))
        gnss_trans = gt_trans
    try:
        mag_tf = tf_buffer.lookup_transform(general_config['tf_frame_world'], general_config['tf_frame_magneto'],
                                            rospy.Time(0))
        mag_rot = np.array([mag_tf.transform.rotation.x, mag_tf.transform.rotation.y, mag_tf.transform.rotation.z,
                            mag_tf.transform.rotation.w])
        mag_rot_euler = np.array(tft.euler_from_quaternion(mag_rot))
    except Exception as e:
        log.log('Magnetometer transform lookup error. {}'.format(e.message))
        mag_rot = gt_rot
        mag_rot_euler = gt_rot_euler

    # obtain gnss, magnetometer errors
    gnss_error = gnss_trans - gt_trans
    gnss_error_abs = np.abs(gnss_error)
    gnss_error_euc = tft.vector_norm(gnss_error)
    mag_error = np.array(tft.quaternion_multiply(tft.quaternion_conjugate(gt_rot), mag_rot))
    mag_error_euler = np.array(tft.euler_from_quaternion(mag_error))
    mag_error_euler_abs = np.abs(mag_error_euler)

    # publish error msg
    error_msg = Error()
    error_msg.header.stamp = rospy.Time.from_sec(timestamp)
    error_msg.trans = list(trans)                                           # estimated translation [east, north, up]
    error_msg.rot = list(rot_euler * 180 / np.pi)                           # estimated rotation [roll, pitch, yaw] in degrees
    error_msg.gt_trans = list(gt_trans)                                     # ground truth translation [east, north, up]
    error_msg.gt_rot = list(gt_rot_euler * 180 / np.pi)                     # ground truth rotation [roll, pitch, yaw] in degrees
    error_msg.gnss_trans = list(gnss_trans)                                 # gnss translation [east, north, up]
    error_msg.magnetometer_rot = list(mag_rot_euler * 180 / np.pi)          # magnetometer rotation [roll, pitch, yaw] in degrees
    error_msg.trans_error = list(error_trans)                               # translational error in estimate [east, north, up]
    error_msg.rot_error = list(error_rot_euler * 180 / np.pi)               # rotational error in estimate [roll, pitch, yaw] in degrees
    error_msg.gnss_error = list(gnss_error)                                 # translational error in gnss [east, north, up]
    error_msg.magnetometer_error = list(mag_error_euler * 180 / np.pi)      # rotational error in magnetometer [roll, pitch, yaw] in degrees
    error_msg.trans_euclidean_error = [error_trans_euc]                     # euclidean translational error in estimate
    error_msg.gnss_euclidean_error = [gnss_error_euc]                       # euclidean translational error in estimate
    error_msg.trans_abs_error = list(error_trans_abs)                       # absolute translational error in estimate [north, east, up]
    error_msg.gnss_abs_error = list(gnss_error_abs)                         # absolute translational error in gnss [north, east, up]
    error_msg.magnetometer_abs_error  = list(mag_error_euler_abs * 180 / np.pi)  # absolute rotational error in magnetometer [roll, pitch, yaw] in degrees
    error_msg.trans_error_upperbound = list(trans_sigma * 3)                # estimated translational upperbounds (3-sigma) [east, north, up]
    error_msg.trans_error_lowerbound = list(-trans_sigma * 3)               # estimated translational lowerbounds (3-sigma) [east, north, up]
    error_msg.rot_error_upperbound = list(rot_sigma * 3 * 180 / np.pi)      # estimated rotational upperbounds (3-sigma) [roll, pitch, yaw] in degrees
    error_msg.rot_error_lowerbound = list(-rot_sigma * 3 * 180 / np.pi)     # estimated rotational lowerbounds (3-sigma) [roll, pitch, yaw] in degrees
    error_publisher.publish(error_msg)

    # broadcast ground truth tf frame
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(timestamp)
    transform.header.frame_id = general_config['tf_frame_world']
    transform.child_frame_id = general_config['tf_frame_groundtruth']
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = list(gt_trans)
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = list(gt_rot)
    tf_broadcaster.sendTransform(transform)

    # publish paths
    publish_path(gt_path, gt_path_publisher, timestamp, general_config['tf_frame_world'], gt_trans, gt_rot)
    publish_path(state_path, state_path_publisher, timestamp, general_config['tf_frame_world'], trans, rot)
    publish_covariance(timestamp, trans_sigma)


if __name__ == '__main__':
    general_config = get_config_dict()['general']
    log = Log(general_config['evaluator_node_name'])
    rospy.init_node(general_config['evaluator_node_name'], anonymous=True)
    log.log('Node initialized.')

    # load ground truth
    ds = KAISTData()
    ds.load_data(imu=False, gnss=False, altitude=False, vlpleft=False, calibrations=False)
    log.log('Ground truth loaded.')

    rospy.Subscriber(general_config['state_topic'], State, callback, queue_size=1)  # subscribe to locator output
    tf_buffer = tf2.Buffer()
    tf_listener = tf2.TransformListener(tf_buffer)

    error_publisher = rospy.Publisher(general_config['error_topic'], Error, queue_size=1)
    gt_path_publisher = rospy.Publisher(general_config['gt_path_topic'], Path, queue_size=1)
    state_path_publisher = rospy.Publisher(general_config['state_path_topic'], Path, queue_size=1)
    cov_ellipsoid_publisher = rospy.Publisher(general_config['cov_ellipsoid_topic'], Marker, queue_size=1)
    tf_broadcaster = tf2.TransformBroadcaster()

    log.log('Node ready.')
    rospy.spin()
