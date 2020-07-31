#!/usr/bin/env python

import rospy
import tf.transformations as tft
import tf
import tf2_ros
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from kalani_v1.msg import State

import numpy as np
import scipy
from scipy.optimize import leastsq
from constants import Constants
from filter.kalman_filter_v1 import Kalman_Filter_V1
from datasetutils.nclt_data_conversions import NCLTDataConversions


nclt_gnss_var = [25.0, 25.0, 100]
nclt_loam_var = 0.001 * np.ones(3)
nclt_mag_orientation_var = 0.001 * np.ones(3)

aw_var = 0.0
ww_var = 0.0

am_var = 0.001
wm_var = 0.001

g = np.array([0, 0, -9.8])

# kf = Kalman_Filter_V1(g, aw_var, ww_var)
kf = Kalman_Filter_V1(g, aw_var, ww_var)

# Latest acceleration measured by the IMU, to be used in estimating orientation in mag_callback()
measured_acceleration = np.zeros(3)


frame1 = '/world'
frame2 = '/loam_origin'


def log(message):
    rospy.loginfo(Constants.LOCATOR_NODE_NAME + ' := ' + str(message))


def publish_state(pub):
    state = kf.get_state_as_numpy()
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(state[-2])
    msg.header.frame_id = Constants.WORLD_FRAME
    msg.position.x, msg.position.y, msg.position.z = list(state[0:3])
    msg.velocity.x, msg.velocity.y, msg.velocity.z = list(state[3:6])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = list(state[6:10])
    [msg.euler.x, msg.euler.y, msg.euler.z] = tft.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], axes='sxyz')
    msg.accleration_bias.x, msg.accleration_bias.y, msg.accleration_bias.z = list(state[10:13])
    msg.angularvelocity_bias.x, msg.angularvelocity_bias.y, msg.angularvelocity_bias.z = list(state[13:16])
    msg.covariance = kf.get_covariance_as_numpy().tolist()
    msg.is_initialized = bool(state[-1])
    pub.publish(msg)
    br.sendTransform(state[0:3],(state[7],state[8],state[9],state[6]),rospy.Time.from_sec(state[-2]),Constants.BODY_FRAME,Constants.WORLD_FRAME)
    # br.sendTransform([0,0,0.5],(state[7],state[8],state[9],state[6]),rospy.Time.from_sec(state[-2]),Constants.BODY_FRAME,Constants.WORLD_FRAME)


def publish_gnss(pub, time, fix):
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.position.x, msg.position.y, msg.position.z = list(fix)
    pub.publish(msg)
    br.sendTransform(fix,(0,0,0,1),rospy.Time.from_sec(time),Constants.GNSS_FRAME,Constants.WORLD_FRAME)


def publish_magnetic(pub, ori, time):
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = ori
    [msg.euler.x, msg.euler.y, msg.euler.z] = tft.euler_from_quaternion(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], axes='sxyz')
    pub.publish(msg)


def publish_mag_orientation(time, q):
    euler = tft.euler_from_quaternion([q[1],q[2],q[3],q[0]],axes='sxyz')
    pub = rospy.Publisher('conv_data/_mag_ori', State, queue_size=1)
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.euler.x, msg.euler.y, msg.euler.z = list(euler)
    pub.publish(msg)


def get_orientation_from_magnetic_field(mm, fm):
    me = np.array([0, -1, 0])
    ge = np.array([0, 0, -1])

    fs = fm / np.linalg.norm(fm)
    ms = mm - np.dot(mm, fs) * fs
    ms = ms / np.linalg.norm(ms)

    def eqn(p):
        x, y, z, w = p
        R = tft.quaternion_matrix([x,y,z,w])[0:3,0:3]
        M = R.dot(ms) + me
        G = R.dot(fs) + ge
        E = np.concatenate((M,G,[w**2+x**2+y**2+z**2-1]))
        return E
    x, y, z, w = scipy.optimize.leastsq(eqn, tft.quaternion_from_euler(0,0,np.pi/4,axes='sxyz'))[0]

    quat_array = tft.unit_vector([w, x, y, z])
    if quat_array[0] < 0: quat_array = -quat_array

    return quat_array


def gnss_callback(data):
    # print 'gps callback'
    time = data.header.stamp.to_sec()
    fix_mode = data.status.status
    lat = data.latitude
    long = data.longitude
    alt = data.altitude
    track = 0
    speed = 0
    gnss_array = np.array([time*1e6,fix_mode,4,lat,long,alt,track,speed])
    gnss = NCLTDataConversions.gnss_numpy_to_converted(gnss_array)
    fix = np.array([gnss.x,gnss.y,gnss.z])
    if gnss.fix_mode == 3:
        if kf.is_initialized():
            Hx = np.zeros([3, 16])
            Hx[:, 0:3] = np.eye(3)
            V = np.diag(nclt_gnss_var)
            kf.correct(fix, Hx, V, time, measurementname='gnss')
            publish_state(state_pub)
            publish_gnss(converted_gnss_pub, time, fix)

        else:
            p = fix
            cov_p = nclt_gnss_var

            v = np.zeros(3)
            cov_v = [0,0,0]

            ab = np.array([0.0, 0.0, 0.0])
            cov_ab = aw_var * np.ones(3)

            wb = np.zeros(3)
            cov_wb = ww_var * np.ones(3)

            t = time

            kf.initialize_state(p=p,cov_p=cov_p,v=v,cov_v=cov_v,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,time=t)

            tf_quat = tft.quaternion_from_euler(0,0,0)
            try:
                (trans, rot) = ls.lookupTransform(frame2, frame1, rospy.Time(0))
                br.sendTransform(p, rot, rospy.Time.from_sec(time), frame2, frame1)
                print 'loam_origin set'
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                br.sendTransform(p, tf_quat, rospy.Time.from_sec(time), frame2, frame1)

    elif gnss.fix_mode == 2:
        if kf.is_initialized():
            Hx = np.zeros([2, 16])
            Hx[:, 0:2] = np.eye(2)
            V = np.diag(nclt_gnss_var[0:2])
            kf.correct(fix[0:2], Hx, V, time, measurementname='gnss_no_alt')
            publish_state(state_pub)
            publish_gnss(converted_gnss_pub, time, fix)

    # print 'gps callback end'


def imu_callback(data):
    # print 'imu callback'

    am = NCLTDataConversions.vector_ned_to_enu(np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))
    global measured_acceleration
    measured_acceleration = am

    am = np.array([am[0],am[1],am[2]])

    wm = NCLTDataConversions.vector_ned_to_enu(np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z]))

    time = data.header.stamp.to_sec()

    if kf.is_initialized():
        kf.predict(am, am_var, wm, wm_var, time, inputname='imu')
        publish_state(state_pub)

    # print 'imu callback end'


def mag_callback(data):
    # print 'mag callback'
    mm = NCLTDataConversions.vector_ned_to_enu(np.array([data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z]))
    time = data.header.stamp.to_sec()

    ori = get_orientation_from_magnetic_field(mm,measured_acceleration)

    if kf.is_initialized():
        Hx = np.zeros([4,16])
        Hx[:,6:10] = np.eye(4)
        V = np.diag(np.ones(4) * 0.0001)
        kf.correct(ori,Hx,V,time, measurementname='magnetometer')
        publish_state(state_pub)
        publish_magnetic(magnetic_pub, ori, time)

        # try:
        #     (trans, rot) = ls.lookupTransform('/loam_origin', '/world', rospy.Time(0))
        #     br.sendTransform(trans, rot, rospy.Time.from_sec(time), '/world', '/loam_origin')
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     pass

    else:
        v = np.zeros(3)
        cov_v = [0, 0, 0]

        q = ori
        cov_q = nclt_mag_orientation_var

        ab = np.zeros(3)
        cov_ab = aw_var * np.ones(3)

        wb = np.zeros(3)
        cov_wb = ww_var * np.ones(3)

        kf.initialize_state(v=v,cov_v=cov_v,q=q,cov_q=cov_q,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,g=g,time=time)

        # tf_quat = np.concatenate([q[1:4],[q[0]]])
        # try:
        #     (trans, rot) = ls.lookupTransform('/loam_origin', '/world', rospy.Time(0))
        #     br.sendTransform(trans, tf_quat, rospy.Time.from_sec(time), '/world', '/loam_origin')
        #     print 'loam_origin set'
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     br.sendTransform((0,0,0), tf_quat, rospy.Time.from_sec(time), '/world', '/loam_origin')

        if kf.is_initialized():
            log('State initialized.')
    # print 'mag callback end'


def loam_callback(data):
        time = data.header.stamp.to_sec()

        p = NCLTDataConversions.vector_ned_to_enu(np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z]))

        q = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        euler_in_ned = tft.euler_from_quaternion(q, axes='sxyz')
        euler_in_enu = NCLTDataConversions.vector_ned_to_enu(euler_in_ned)
        q = tft.quaternion_from_euler(euler_in_enu[0], euler_in_enu[1], euler_in_enu[2], axes='sxyz')

        print 'loam p, q:', p, q

        br.sendTransform(p,q,rospy.Time.from_sec(time), '/loam_link', '/loam_origin')

if __name__ == '__main__':
    rospy.init_node(Constants.LOCATOR_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber(Constants.NCLT_RAW_DATA_GNSS_FIX_TOPIC, NavSatFix, gnss_callback, queue_size=1)
    rospy.Subscriber(Constants.NCLT_RAW_DATA_IMU_TOPIC, Imu, imu_callback, queue_size=1)
    rospy.Subscriber(Constants.NCLT_RAW_DATA_MAGNETOMETER_TOPIC, MagneticField, mag_callback, queue_size=1)
    rospy.Subscriber('/aft_mapped_to_init', Odometry, loam_callback, queue_size=1)

    state_pub = rospy.Publisher(Constants.STATE_TOPIC, State, queue_size=1)
    converted_gnss_pub = rospy.Publisher(Constants.CONVERTED_GNSS_DATA_TOPIC, State, queue_size=1)
    magnetic_pub = rospy.Publisher(Constants.CONVERTED_MAGNETIC_DATA_TOPIC, State, queue_size=1)

    br = tf.TransformBroadcaster()
    ls = tf.TransformListener()
    br_static = tf2_ros.StaticTransformBroadcaster()
    log('Locator ready.')
    rospy.spin()
