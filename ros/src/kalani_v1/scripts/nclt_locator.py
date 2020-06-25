#!/usr/bin/env python

import rospy
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import NavSatStatus
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from kalani_v1.msg import State

import numpy as np
import scipy
from scipy.optimize import leastsq

from constants import Constants
from filter.kalman_filter_v1 import Kalman_Filter_V1
from filter.rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


kf = Kalman_Filter_V1()


nclt_gnss_var = [10.0, 10.0, 200]
nclt_imu_acceleration_var = [0.001, 0.001, 0.001]
nclt_imu_angularvelocity_var = [0.001, 0.001, 0.001]
nclt_imu_acceleration_bias_var = [0.001, 0.001, 0.001]
nclt_imu_angularvelocity_bias_var = [0.001, 0.001, 0.001]
nclt_mag_orientation_var = [1,1,1]


# Rotation matrix from NED to ENU
R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

# Latest acceleration measured by the IMU, to be used in estimating orientation in mag_callback()
measured_acceleration = np.zeros(3)


def log(message):
    rospy.loginfo(Constants.FILTER_NODE_NAME + ' := ' + str(message))


def publish_state():
    state = kf.get_state_as_numpy()
    pub = rospy.Publisher(Constants.STATE_TOPIC, State, queue_size=1)
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(state[0])
    msg.header.frame_id = Constants.STATE_FRAME
    msg.position.x, msg.position.y, msg.position.z = list(state[1:4])
    msg.velocity.x, msg.velocity.y, msg.velocity.z = list(state[4:7])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = list(state[7:11])
    msg.accleration_bias.x, msg.accleration_bias.y, msg.accleration_bias.z = list(state[11:14])
    msg.angularvelocity_bias.x, msg.angularvelocity_bias.y, msg.angularvelocity_bias.z = list(state[14:17])
    msg.covariance = kf.get_covariance_as_numpy().tolist()
    msg.is_initialized = kf.state_initialized
    pub.publish(msg)

def publish_gnss(fix):
    pub = rospy.Publisher('converted_gnss', State, queue_size=1)
    msg = State()
    msg.position.x, msg.position.y, msg.position.z = list(fix)
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
    fix_mode = data.status.status
    fix = np.deg2rad(np.array([data.latitude, data.longitude, data.altitude]))
    time = data.header.stamp.to_sec()

    origin = np.array([42.293227 * np.pi / 180, -83.709657 * np.pi / 180, 270])
    dif = fix - origin

    # GNSS data in NED
    r = 6400000
    fix[0] = r * np.sin(dif[0])
    fix[1] = r * np.cos(origin[0]) * np.sin(dif[1])
    fix[2] = dif[2]

    # GNSS data in ENU
    fix = np.matmul(R_ned_enu,fix)

    if fix_mode == NavSatStatus.STATUS_FIX and fix[2]!='nan':
        if kf.state_initialized:
            Hx = np.zeros([3, 16])
            Hx[:, 0:3] = np.eye(3)
            V = np.diag(nclt_gnss_var)
            kf.correct(fix, Hx, V, time)
            publish_state()
            publish_gnss(fix)

        else:
            p = fix
            cov_p = nclt_gnss_var

            v = np.zeros(3)
            cov_v = [0,0,0]

            ab = np.zeros(3)
            cov_ab = nclt_imu_acceleration_bias_var

            wb = np.zeros(3)
            cov_wb = nclt_imu_angularvelocity_bias_var

            g = np.array([0, 0, -9.8])

            t = time

            kf.initialize_state(p=p,cov_p=cov_p,v=v,cov_v=cov_v,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,g=g,time=t)

            if kf.state_initialized:
                log('State initialized.')


def imu_callback(data):
    am = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
    am = np.matmul(R_ned_enu, am)

    global measured_acceleration
    measured_acceleration = am

    wm = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
    wm = np.matmul(R_ned_enu, wm)

    time = data.header.stamp.to_sec()

    if kf.state_initialized:
        kf.predict(am,wm,time)
        publish_state()


def mag_callback(data):
    mm = np.array([data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z])
    mm = np.matmul(R_ned_enu, mm)
    time = data.header.stamp.to_sec()

    if kf.state_initialized:
        pass
    else:
        v = np.zeros(3)
        cov_v = [0, 0, 0]

        q = get_orientation_from_magnetic_field(mm,measured_acceleration)
        cov_q = nclt_mag_orientation_var

        ab = np.zeros(3)
        cov_ab = nclt_imu_acceleration_bias_var

        wb = np.zeros(3)
        cov_wb = nclt_imu_angularvelocity_bias_var

        g = np.array([0, 0, -9.8])

        kf.initialize_state(v=v,cov_v=cov_v,q=q,cov_q=cov_q,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,g=g,time=time)

        if kf.state_initialized:
            log('State initialized.')


if __name__ == '__main__':
    rospy.init_node(Constants.FILTER_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber(Constants.NCLT_RAW_DATA_GNSS_FIX_TOPIC, NavSatFix, gnss_callback, queue_size=1)
    rospy.Subscriber(Constants.NCLT_RAW_DATA_IMU_TOPIC, Imu, imu_callback, queue_size=1)
    rospy.Subscriber(Constants.NCLT_RAW_DATA_MAGNETOMETER_TOPIC, MagneticField, mag_callback, queue_size=1)
    rospy.spin()
