#!/usr/bin/env python2

import rospy
import tf
from kalani_v1.msg import State
from kalani_v1.msg import Error
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
import visualization_msgs

import numpy as np
from scipy.interpolate import interp1d
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, Quaternion
from constants import Constants
from datasetutils.kaist_data_conversions import KAISTDataConversions, KAISTData

gt = []
gt_function = []
pos_cal = np.zeros((3, 3))
ori_cal = np.zeros((3, 3))
t = np.zeros(1)

groundtruth_path = Path()
estimate_path = Path()


def log(message):
    rospy.loginfo(Constants.EVALUATOR_NODE_NAME + ' := ' + str(message))


def publish_covariance(data, publisher, frameid, threesigma):
    marker = Marker()
    marker.header.frame_id = frameid
    marker.header.stamp = data.header.stamp
    marker.ns = 'covariances'
    marker.type = visualization_msgs.msg.Marker.SPHERE
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.pose.position = data.position
    marker.pose.orientation = data.orientation

    marker.scale.x = threesigma[0] * 2
    marker.scale.y = threesigma[1] * 2
    marker.scale.z = threesigma[2] * 2
    marker.color.a = 0.5
    marker.color.r = 100.0
    marker.color.g = 100.0
    marker.color.b = 0.0
    publisher.publish(marker)


def publish_path(path, data, publisher, frameid):
    path.header.stamp = data.header.stamp
    path.header.frame_id = frameid
    pose = PoseStamped()
    pose.header.stamp = data.header.stamp
    pose.header.frame_id = frameid
    pose.pose.position = data.position
    pose.pose.orientation = data.orientation
    path.poses.append(pose)
    publisher.publish(path)


def publish_error(pub):
    msg = Error()
    msg.header.stamp = rospy.Time.from_sec(t[0])
    msg.position.x, msg.position.y, msg.position.z = pos_cal[0:3, 0]
    msg.x_eb_u = [pos_cal[0, 1]]
    msg.x_eb_d = [pos_cal[0, 2]]
    msg.y_eb_u = [pos_cal[1, 1]]
    msg.y_eb_d = [pos_cal[1, 2]]
    msg.z_eb_u = [pos_cal[2, 1]]
    msg.z_eb_d = [pos_cal[2, 2]]

    msg.orientation.x, msg.orientation.y, msg.orientation.z = ori_cal[0:3, 0]
    msg.r_eb_u = [ori_cal[0, 1]]
    msg.p_eb_u = [ori_cal[1, 1]]
    msg.ya_eb_u = [ori_cal[2, 1]]
    msg.r_eb_d = [ori_cal[0, 2]]
    msg.p_eb_d = [ori_cal[1, 2]]
    msg.ya_eb_d = [ori_cal[2, 2]]

    pub.publish(msg)


def publish_gt(pub, time, position, ori_e):
    ori_q = Quaternion(euler=ori_e).to_numpy()
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.position.x, msg.position.y, msg.position.z = list(position)
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = list(ori_q)
    msg.euler.x, msg.euler.y, msg.euler.z = list(ori_e)
    pub.publish(msg)
    publish_path(groundtruth_path, msg,gt_path_pub,'odom')
    br.sendTransform(position, (ori_q[1],ori_q[2],ori_q[3],ori_q[0]), rospy.Time.from_sec(time), 'gt', 'odom')
    br.sendTransform((0, 10, 0), (ori_q[1],ori_q[2],ori_q[3],ori_q[0]), rospy.Time.from_sec(time), 'gt_ori', 'odom')


def state_callback(data):

    # getting data from state topic
    state = np.array([data.header.stamp.to_sec(), data.position.x, data.position.y, data.position.z, data.orientation.w, data.orientation.x,
                      data.orientation.y, data.orientation.z])
    cov_est = data.covariance

    # Quaternion transformation

    p_est_euler = []
    p_cov_euler_std = []
    p_cov_std = []

    qc = Quaternion(state[4], state[5], state[6], state[7]).normalize()
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())

    cal_mat1 = [cov_est[96:99], cov_est[111:114], cov_est[126:129]]
    cal = np.dot(J, cal_mat1)
    p_cov_euler_std.append(np.sqrt(np.diagonal(np.dot(cal, J.T))))

    # Covariance for position
    cal_mat2 = [cov_est[0:3], cov_est[15:18], cov_est[30:33]]
    p_cov_std.append(np.sqrt(np.diagonal(cal_mat2)))

    p_est_euler = np.array(p_est_euler)
    p_cov_euler_std = np.array(p_cov_euler_std)
    p_cov_std = np.array(p_cov_std)

    # error calculation

    gt_interpol = gt_function(state[0])

    print state[0], '%d'%gt[0,0]

    t[0] = state[0]

    # position error calculation
    for i in range(3):
        pos_cal[i, 0] = gt_interpol[i] - state[i + 1]
        pos_cal[i, 1] = 3 * p_cov_std[:, i]
        pos_cal[i, 2] = -3 * p_cov_std[:, i]

    # orientation error calculation
    for i in range(3):
        ori_cal[i, 0] = angle_normalize(gt_interpol[i + 3] - p_est_euler[:, i])
        ori_cal[i, 1] = 3 * p_cov_euler_std[:, i]
        ori_cal[i, 2] = -3 * p_cov_euler_std[:, i]

    publish_error(error_pub)
    publish_gt(gt_pub, state[0],gt_interpol[0:3], gt_interpol[3:6])
    publish_path(estimate_path, data, et_path_pub, 'odom')
    publish_covariance(data, cov_ellipse_pub, 'odom', pos_cal[:,1])


def gnss_callback(data):
    time = data.header.stamp.to_sec()
    fix = np.array([data.position.x, data.position.y, data.position.z])
    gt_interpol = gt_function(time)
    msg = Error()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.position.x, msg.position.y, msg.position.z = (gt_interpol[0:3] - fix).tolist()
    gnss_error_pub.publish(msg)


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.EVALUATOR_NODE_NAME, anonymous=True)
        log('Node initialized.')

        ds = KAISTData('/home/pasan/kalani-data/kaist/urban17', gnss=False)
        gt = np.array([ds.groundtruth.time, ds.groundtruth.x, ds.groundtruth.y, ds.groundtruth.z, ds.groundtruth.r, ds.groundtruth.p, ds.groundtruth.h]).T

        # gt_function = interp1d(gt[:, 0], gt[:, 1:7], axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
        gt_function = interp1d(gt[:, 0], gt[:, 1:7], axis=0, bounds_error=False, fill_value=gt[0, 1:7], kind='linear')
        log('Interpolation finished.')

        rospy.Subscriber(Constants.STATE_TOPIC, State, state_callback, queue_size=1)
        rospy.Subscriber(Constants.CONVERTED_GNSS_DATA_TOPIC, State, gnss_callback, queue_size=1)
        # rospy.Subscriber(Constants.CONVERTED_MAGNETIC_DATA_TOPIC, State, magnetic_callback, queue_size=1)

        br = tf.TransformBroadcaster()
        gt_pub = rospy.Publisher(Constants.CONVERTED_GROUNDTRUTH_DATA_TOPIC, State, queue_size=1)
        error_pub = rospy.Publisher(Constants.ERROR_TOPIC, Error, queue_size=1)
        gnss_error_pub = rospy.Publisher('gnss_error', Error, queue_size=1)
        et_path_pub = rospy.Publisher('estimate_path', Path, queue_size=10)
        gt_path_pub = rospy.Publisher('groundtruth_path', Path, queue_size=10)
        cov_ellipse_pub = rospy.Publisher('cov_ellipse', Marker, queue_size=10)

        log('Evaluator ready.')
        rospy.spin()
    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
