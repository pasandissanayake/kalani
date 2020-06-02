#!/usr/bin/env python

import rospy
from kalani_v1.msg import IMU
from kalani_v1.msg import GNSS
from kalani_v1.msg import State

import numpy as np
import scipy
from scipy.optimize import leastsq

from constants import Constants
from filter.filter_v1 import Filter_V1
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


kf = Filter_V1()


def log(message):
    rospy.loginfo(Constants.FILTER_NODE_NAME + ' := ' + str(message))


def publish_state():
    state = kf.get_state_as_numpy()
    pub = rospy.Publisher(Constants.STATE_TOPIC, State, queue_size=10)
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(state[0])
    msg.header.frame_id = Constants.STATE_FRAME
    msg.position.x, msg.position.y, msg.position.z = list(state[1:4])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = list(state[7:11])
    msg.covariance = kf.get_covariance_as_numpy().tolist()
    msg.is_initialized = kf.state_initialized
    pub.publish(msg)


def get_orientation_from_magnetic_field(mm, am):
    me = np.array([0, -1, 0])
    ge = np.array([0, 0, -1])
    def eqn(p):
        w, x, y, z = p
        R = Quaternion(w,x,y,z).normalize().to_mat()
        M = R.dot(mm) + me
        G = R.dot(am) + ge
        E = np.concatenate((M,G,[w**2+x**2+y**2+z**2-1]))
        return E

    w, x, y, z = scipy.optimize.leastsq(eqn, Quaternion(euler=[0, 0, np.pi / 4]).to_numpy())[0]

    return Quaternion(w,x,y,z).normalize().to_numpy()


def gnss_callback(data):
    # Rotation matrix from NED to ENU
    R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    # time, fix(2=w/o alt, 3=with alt), lat, long, alt
    gnss = np.array([data.header.stamp.to_sec(),data.fix_mode,data.latitude,data.longitude,data.altitude])
    origin = np.array([42.293227 * np.pi / 180, -83.709657 * np.pi / 180, 270])
    dif = gnss[2:5] - origin

    # GNSS data in NED
    r = 6400000
    gnss[2] = r * np.sin(dif[0])
    gnss[3] = r * np.cos(origin[0]) * np.sin(dif[1])
    gnss[4] = dif[2]

    # GNSS data in ENU
    gnss[2:5] = (R_ned_enu.dot(gnss[2:5].T)).T

    if kf.state_initialized:
        if gnss[1] == 3:
            kf.correct(gnss[2:5],gnss[0],Filter_V1.GNSS_WITH_ALT)
        elif gnss[1] == 2:
            kf.correct(gnss[2:4],gnss[0],Filter_V1.GNSS_NO_ALT)
        publish_state()
    else:
        if gnss[1] == 3:
            p = gnss[2:5]
            cov_p = [kf.var_gnss_with_alt_horizontal,kf.var_gnss_with_alt_horizontal,kf.var_gnss_with_alt_vertical]

            v = np.zeros(3)
            cov_v = [0,0,0]

            ab = np.zeros(3)
            cov_ab = [kf.var_imu_aw,kf.var_imu_aw,kf.var_imu_aw]

            wb = np.zeros(3)
            cov_wb = [kf.var_imu_ww,kf.var_imu_ww,kf.var_imu_ww]

            g = np.array([0, 0, -9.8])

            t = gnss[0]

            kf.initialize_state(p=p,cov_p=cov_p,v=v,cov_v=cov_v,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,g=g,time=t)

            if kf.state_initialized:
                log('State initialized.')


def imu_callback(data):
    mm = np.array([data.magnetic_field.x,data.magnetic_field.y,data.magnetic_field.z])
    am = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
    wm = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
    t = data.header.stamp.to_sec()

    if kf.state_initialized:
        kf.predict(am,wm,t)
        publish_state()
    else:
        v = np.zeros(3)
        cov_v = [0, 0, 0]

        # q = get_orientation_from_magnetic_field(mm,am)
        q = np.array([1,0,0,0])
        cov_q = [1,1,1]

        ab = np.zeros(3)
        cov_ab = [kf.var_imu_aw, kf.var_imu_aw, kf.var_imu_aw]

        wb = np.zeros(3)
        cov_wb = [kf.var_imu_ww, kf.var_imu_ww, kf.var_imu_ww]

        g = np.array([0, 0, -9.8])

        kf.initialize_state(v=v,cov_v=cov_v,q=q,cov_q=cov_q,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,g=g,time=t)

        if kf.state_initialized:
            log('State initialized.')


if __name__ == '__main__':
    rospy.init_node(Constants.FILTER_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber(Constants.GNSS_DATA_TOPIC, GNSS, gnss_callback)
    rospy.Subscriber(Constants.IMU_DATA_TOPIC, IMU, imu_callback)
    rospy.spin()