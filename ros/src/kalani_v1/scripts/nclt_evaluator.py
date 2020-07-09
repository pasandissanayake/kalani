#!/usr/bin/env python

import rospy
import tf

from kalani_v1.msg import State
from kalani_v1.msg import Error
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
import visualization_msgs

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

from constants import Constants

df = pd.read_csv(Constants.NCLT_GROUNDTRUTH_DATA_PATH, header=None)

gt = []
gt_function = []
pos_cal = np.zeros((3, 3))
ori_cal = np.zeros((3, 3))
t = np.zeros(1)


def log(message):
    rospy.loginfo(Constants.EVALUATOR_NODE_NAME + ' := ' + str(message))


def publish_covariance(data, publisher, frameid, covariance):
    marker = Marker()
    marker.header.frame_id = frameid
    marker.header.stamp = data.header.stamp
    marker.ns = "my_namespace2"
    marker.type = visualization_msgs.msg.Marker.SPHERE
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.pose.position = data.position
    marker.pose.orientation = data.orientation

    marker.scale.x = covariance[0]
    marker.scale.y = covariance[1]
    marker.scale.z = covariance[2]
    marker.color.a = 1.0
    marker.color.r = 100.0
    marker.color.g = 100.0
    marker.color.b = 0.0
    publisher.publish(marker)


def publish_path(data, publisher, frameid):
    path = Path()
    path.header.stamp = data.header.stamp
    path.header.frame_id = frameid
    pose = PoseStamped()
    pose.header.stamp = data.header.stamp
    pose.header.frame_id = frameid
    pose.pose.position = data.position
    pose.pose.orientation = data.orientation
    path.poses.append(pose)
    publisher.publish(path)


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

    publish_error()
    publish_gt(state[0],gt_interpol[0:3], gt_interpol[3:6])
    publish_path(data, et_path_pub, 'world')
    publish_covariance(data, cov_ellipse_pub, 'world', pos_cal[:,1])


def publish_error():
    pub = rospy.Publisher(Constants.ERROR_TOPIC, Error, queue_size=1)
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

def publish_gt(time, position, ori_e):
    pub = rospy.Publisher('converted_gt', State, queue_size=1)
    ori_q = Quaternion(euler=ori_e).to_numpy()
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.position.x, msg.position.y, msg.position.z = list(position)
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = list(ori_q)
    msg.euler.x, msg.euler.y, msg.euler.z = list(ori_e)
    pub.publish(msg)
    publish_path(msg,gt_path_pub,'world')
    br.sendTransform(position, (ori_q[1],ori_q[2],ori_q[3],ori_q[0]), rospy.Time.from_sec(time), 'gt', 'world')


def imu_callback(data):
    global imu_time
    imu_time = data.header.stamp.to_sec()


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.EVALUATOR_NODE_NAME, anonymous=True)
        log('Node initialized.')

        i_gtruth = 0
        gtruth_input = []
        while not rospy.is_shutdown() and len(df) > i_gtruth:
            gtruth_input.append(list(df.loc[i_gtruth]))
            if i_gtruth <= (len(df) - 1):
                i_gtruth = i_gtruth + 1
            else:
                break
        gtruth_input = np.array(gtruth_input)

        R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        gt = np.zeros(gtruth_input.shape)
        gt[:, 0] = gtruth_input[:, 0] * 10 ** (-6)
        for i in range(len(gt)):
            gt[i,1:4] = np.matmul(R_ned_enu, gtruth_input[i,1:4])
            gt[i, 4:7] = np.matmul(R_ned_enu, gtruth_input[i, 4:7])

        log('Data storing finished.')
        gt = np.array(gt)

        gt_function = interp1d(gt[:, 0], gt[:, 1:7], axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
        log('Interpolation finished.')

        rospy.Subscriber(Constants.STATE_TOPIC, State, state_callback, queue_size=1)
        br = tf.TransformBroadcaster()
        et_path_pub = rospy.Publisher('estimate_path', Path, queue_size=10)
        gt_path_pub = rospy.Publisher('groundtruth_path', Path, queue_size=10)
        cov_ellipse_pub = rospy.Publisher('cov_ellipse', Marker, queue_size=10)
        log('Evaluator ready.')
        rospy.spin()
    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')
