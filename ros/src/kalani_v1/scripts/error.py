#!/usr/bin/env python

import rospy

from kalani_v1.msg import State
from kalani_v1.msg import Error
from std_msgs.msg import MultiArrayLayout

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

from constants import Constants

df = pd.read_csv(Constants.GROUNDTRUTH_DATA_PATH, header=None)

state_initialized = False
gt = []
pos_cal = np.zeros((3, 3))
ori_cal = np.zeros((3, 3))
t = np.zeros((1))


def log(message):
    rospy.loginfo(Constants.FILTER_NODE_NAME + ' := ' + str(message))


def state_callback(data):
    state_initialized = data.is_initialized

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
    gt_function = interp1d(gt[:, 0], gt[:, 1:7], axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
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

    publish_state()


def publish_state():
    pub = rospy.Publisher(Constants.ERROR_TOPIC, Error, queue_size=10)
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


def imu_callback(data):
    global imu_time
    imu_time = data.header.stamp.to_sec()


if __name__ == '__main__':
    try:
        rospy.init_node(Constants.ERROR_CAL_NODE_NAME, anonymous=True)
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
        gt[:, 1:4] = (np.dot(R_ned_enu, gtruth_input[:, 1:4].T)).T

        for i in range(len(gt)):
            q_enu_ned = Quaternion(euler=np.array([0, np.pi, np.pi / 2]))
            q_in_ned = Quaternion(euler=gt[i, 4:7])
            q = q_enu_ned.quat_mult_left(q_in_ned)
            q = Quaternion(q[0],q[1],q[2],q[3])
            gt[i,4:7] = q.to_euler()

        rospy.loginfo(str("data storing finished"))
        gt = np.array(gt)

        rospy.Subscriber(Constants.STATE_TOPIC, State, state_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        log('Stopping node unexpectedly.')