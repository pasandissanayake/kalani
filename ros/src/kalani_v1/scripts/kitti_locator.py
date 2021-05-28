#!/usr/bin/python2

import numpy as np
import numdifftools as numdiff

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from kalani_v1.msg import State
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from kalman_filter import KalmanFilter
from utilities import *
from kaist_datahandle import KAISTData


# placeholders for most recent measurement values
gnss_fix = None
linear_acceleration = None
angular_acceleration = None
magnetic_field = None
altitude = None
var_altitude = None

# placeholders for message sequence numbers
seq_imu = -1
seq_gnss = -1
seq_altimeter = -1

# absolute stationarity condition (detected by odometers)
stationary = True

def is_stationary():
    global stationary
    return stationary


def publish_state(transform_only=False):
    state, timestamp, is_valid = kf.get_current_state()

    if not transform_only:
        msg = State()
        msg.header.stamp = rospy.Time.from_sec(timestamp)
        msg.header.frame_id = config['tf_frame_odom']
        msg.position.x, msg.position.y, msg.position.z = list(state[0:3])
        msg.velocity.x, msg.velocity.y, msg.velocity.z = list(state[3:6])
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = list(state[6:10])
        [msg.euler.x, msg.euler.y, msg.euler.z] = tft.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], axes='sxyz')
        msg.accleration_bias.x, msg.accleration_bias.y, msg.accleration_bias.z = list(state[10:13])
        msg.angularvelocity_bias.x, msg.angularvelocity_bias.y, msg.angularvelocity_bias.z = list(state[13:16])
        msg.covariance = kf.get_current_cov()
        msg.is_initialized = is_valid
        state_pub.publish(msg)

    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(timestamp)
    transform.header.frame_id = config['tf_frame_odom']
    transform.child_frame_id = config['tf_frame_state']
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z  = state[0:3]
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = state[6:10]
    tf2_broadcaster.sendTransform(transform)


def publish_gnss(timestamp, fix):
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(timestamp)
    transform.header.frame_id = config['tf_frame_world']
    transform.child_frame_id = config['tf_frame_gnss']
    transform.transform.translation.x, transform.transform.translation.y = list(fix)
    transform.transform.translation.z = altitude
    transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z = (1, 0, 0, 0)
    tf2_broadcaster.sendTransform(transform)


def publish_magnetic(timestamp, ori):
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(timestamp)
    transform.header.frame_id = config['tf_frame_world']
    transform.child_frame_id = config['tf_frame_magneto']
    transform.transform.translation.x, transform.transform.translation.y = list(gnss_fix)
    transform.transform.translation.z = altitude
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = list(ori)
    tf2_broadcaster.sendTransform(transform)


def gnss_callback(data):
    global gnss_fix, altitude, var_altitude, seq_gnss
    t = data.header.stamp.to_sec()

    # check for message misses
    if data.header.seq > seq_gnss + 1:
        log.log("{} GNSS messages lost at {} s!".format(data.header.seq - seq_gnss -1, t))
    seq_gnss = data.header.seq

    # prepare pseudo-gnss fix
    cov = np.reshape(data.pose.covariance, (6,6))[0:2, 0:2]
    gt_fix = np.array([kd.groundtruth.interp_x(t), kd.groundtruth.interp_y(t), kd.groundtruth.interp_z(t)])
    # gnss_fix = gt_fix[0:2] # exact ground truth as gnss
    gnss_fix = gt_fix[0:2] + np.random.normal(0, np.sqrt(np.max(cov)), (2)) # pseudo-gnss from ground truth
    # gnss_fix = np.array([data.pose.pose.position.x, data.pose.pose.position.y]) # original gnss
        
    if kf.is_initialized:
        # gnss correction
        if not is_stationary():
            # simulate GNSS outage
            # if True and (t > 1544590908.4 - 15.0 and t < 1544590908.4 + 15.0): # urban28
            if False and (t > 1544590943.49 + 5.0 and t < 1544590943.49 + 25.0): 
                log.log('no gps!')
                pass
            elif seq_gnss % 1 == 0:
                def meas_fun(ns):
                    return ns.p()[0:2]
                def hx_fun(ns):
                    return np.concatenate([np.eye(2),np.zeros((2,14))], axis=1)
                kf.correct_absolute(meas_fun, gnss_fix, cov*2, t, hx_fun=hx_fun, measurement_name='gnss')
                pass
        else:
            log.log('gps discarded - stationary')
        publish_gnss(t, gnss_fix)

    elif altitude is not None:
        # filter initialization
        # orientation_gt = tft.quaternion_from_euler(kd.groundtruth.interp_r(t), kd.groundtruth.interp_p(t), kd.groundtruth.interp_h(t)) 
        # cov_q = np.eye(3)*1e-3

        cov_p = np.eye(3)
        cov_p[0:2, 0:2] = cov
        cov_p[2, 2] = var_altitude
        # p = np.concatenate([gnss_fix, [altitude]]) # initialize by gnss
        p = gt_fix # initialize by ground truth
        kf.initialize([
            ['p', p, cov_p, t],
            ['v', init_velocity, np.diag(init_var_velocity), t],
            # ['q', orientation_gt, cov_q, t],
            ['ab', init_imu_linear_acceleration_bias, np.diag(var_imu_linear_acceleration_bias), t],
            ['wb', init_imu_angular_velocity_bias, np.diag(var_imu_angular_velocity_bias), t]
        ])
    

def alt_callback(data):
    global altitude, var_altitude, seq_altimeter
    t = data.header.stamp.to_sec()

    # check for message misses
    if data.header.seq > seq_altimeter + 1:
        log.log("{} Altimeter messages lost at {} s!".format(data.header.seq - seq_altimeter -1, t))
    seq_altimeter = data.header.seq

    # prepare pseudo-altitude measurement
    var_altitude = np.reshape(data.pose.covariance, (6, 6))[2, 2]
    altitude = kd.groundtruth.interp_z(t) + np.random.normal(0, np.sqrt(var_altitude))

    if kf.is_initialized:
        # altitude correction
        def meas_fun(ns):
            return np.array([ns.p()[2]])
        def hx_fun(ns):
            return np.concatenate([[0,0,1], np.zeros(13)]).reshape((1, 16))
        # kf.correct_absolute(meas_fun, np.array([altitude]), var_altitude.reshape((1,1)), t, hx_fun=hx_fun, measurement_name='altitude')


imu_rate_adjust = 1
def imu_callback(data):
    global linear_acceleration, angular_velocity, seq_imu, imu_rate_adjust
    t = data.header.stamp.to_sec()

    # check for message misses
    if data.header.seq > seq_imu + 1:
        log.log("{} IMU messages lost at {} s!".format(data.header.seq - seq_imu -1, t))
    seq_imu = data.header.seq

    if imu_rate_adjust % 1 == 0:
        imu_rate_adjust = 1
    else:
        imu_rate_adjust = imu_rate_adjust + 1
        return

    linear_acceleration = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
    angular_velocity = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
    var_am = np.reshape(data.linear_acceleration_covariance, (3,3))
    var_wm = np.reshape(data.angular_velocity_covariance, (3,3))
    cov = np.eye(12)
    cov[0:3, 0:3] = var_am
    cov[3:6, 3:6] = var_wm
    cov[6:9, 6:9] = var_imu_linear_acceleration_bias
    cov[9:12, 9:12] = var_imu_angular_velocity_bias

    if kf.is_initialized:
        # prediction
        inputs = np.concatenate([linear_acceleration, angular_velocity])
        kf.predict(inputs, cov, t, input_name='imu')
        publish_state()


mag_rate_adjust = 1
def mag_callback(data):
    global magnetic_field, linear_acceleration, mag_rate_adjust

    if mag_rate_adjust % 10 == 0:
        mag_rate_adjust = 1
    else:
        mag_rate_adjust = mag_rate_adjust + 1
        return

    t = data.header.stamp.to_sec()
    magnetic_field = np.array([data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z])
    cov = np.reshape(data.magnetic_field_covariance, (3,3))
    # orientation = get_orientation_from_magnetic_field(magnetic_field, linear_acceleration)
    orientation_gt = tft.quaternion_from_euler(kd.groundtruth.interp_r(t), kd.groundtruth.interp_p(t), kd.groundtruth.interp_h(t)) # for initialization
    orientation = orientation_gt

    angle, axis = quaternion_to_angle_axis(orientation)
    meas_axisangle = angle * axis
    if kf.is_initialized:
        # magnetometer correction
        def meas_fun(ns):
            angle, axis = quaternion_to_angle_axis(ns.q())
            return angle * axis
        def hx_fun(ns):
            [x, y, z, w] = ns.q()
            n = tft.vector_norm([x, y, z])
            if w != 0:
                h = 2 * np.arctan(n / w)
                a = 1 + n**2 / w**2
                p = lambda p1, p2: (2*p1*p2) / (w*a*n**2) - (p1*p2*h)/n**3
                q = np.array([
                    [p(x, x), p(x, y), p(x, z)],
                    [p(y, x), p(y, y), p(y, z)],
                    [p(z, x), p(z, y), p(z, z)],
                ])
                r = -2 * np.array([[x],[y],[z]]) / (a * w**2)
                s = np.zeros((3, 16))
                s[:, 6:10] = np.concatenate([q + np.eye(3) * h/n, r], axis=1)
                return s
            else:
                p = lambda p1, p2: - (p1 * p2 * np.pi) / n ** 3
                q = np.array([
                    [p(x, x), p(x, y), p(x, z)],
                    [p(y, x), p(y, y), p(y, z)],
                    [p(z, x), p(z, y), p(z, z)],
                ])
                r = np.array([[0], [0], [0]])
                s = np.zeros((3, 16))
                s[:, 6:10] = np.concatenate([q + np.eye(3) * np.pi / n, r], axis=1)
                return s
        if not is_stationary():
            # kf.correct_absolute(meas_fun, meas_axisangle, cov, t, hx_fun=hx_fun, measurement_name='magnetometer')
            pass
        publish_magnetic(t, orientation)
    else:
        kf.initialize([
            ['v', init_velocity, np.diag(init_var_velocity), t],
            ['q', orientation_gt, cov, t],
            ['ab', init_imu_linear_acceleration_bias, np.diag(var_imu_linear_acceleration_bias), t],
            ['wb', init_imu_angular_velocity_bias, np.diag(var_imu_angular_velocity_bias), t]
        ])


lo_rate_adjust = 1
def laserodom_callback(data):
    global stationary, lo_rate_adjust
    t1 = data.header.stamp.to_sec()
    t0 = data.twist.twist.linear.z

    # check for validity
    if t0 < 0:
        return

    # delta-position and quaternions in camera-init frame
    dp_ci = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
    ciqc1 = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    ciqc0 = np.array([data.twist.twist.angular.x, data.twist.twist.angular.y, data.twist.twist.angular.z, data.twist.twist.linear.x])
    ciRc0 = tft.quaternion_matrix(ciqc0)[0:3,0:3]
    ciRc1 = tft.quaternion_matrix(ciqc1)[0:3,0:3]

    # obtain vehicle to camera transform
    try:
        trans = tf2_buffer.lookup_transform(config['tf_frame_state'], config['tf_frame_lidar'], rospy.Time(0))
    except:
        log.log('laserodom_callback: transformation error.')
        return
    vqc = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
    vRc = tft.quaternion_matrix(vqc)[0:3,0:3]

    v_ci = dp_ci / (t1-t0) # velocity in camera init frame
    v_c0 = np.matmul(ciRc0.T, v_ci) # velocity in old camera frame
    v_v = np.matmul(vRc, v_c0) * 1.1 # velocity in vehicle frame
    # ZUPT conditions
    v_v[0] = 0
    v_v[2] = 0

    # obtain velocity from ground truth
    # r1 = np.array([kd.groundtruth.interp_r(t1), kd.groundtruth.interp_p(t1), kd.groundtruth.interp_h(t1)])
    # q1 = tft.quaternion_from_euler(r1[0], r1[1], r1[2])
    # p1 = np.array([kd.groundtruth.interp_x(t1), kd.groundtruth.interp_y(t1), kd.groundtruth.interp_z(t1)])
    # p0 = np.array([kd.groundtruth.interp_x(t0), kd.groundtruth.interp_y(t0), kd.groundtruth.interp_z(t0)])
    # dp = p1-p0
    # q_mat = tft.quaternion_matrix(q1)[0:3, 0:3]
    # dp_v = np.dot(q_mat.T, dp)
    # v_vgt = dp_v / (t1-t0)
    
    c0qc1 = tft.quaternion_multiply(tft.quaternion_conjugate(ciqc0), ciqc1)
    v0qv1 = tft.quaternion_multiply(vqc, tft.quaternion_multiply(c0qc1, tft.quaternion_conjugate(vqc)))
    dangle, daxis = quaternion_to_angle_axis(v0qv1)
    dtheta = dangle * daxis # axis angle difference of c1-c0

    # log.log("angle:{}, t0:{}, t1:{}".format(tft.euler_from_quaternion(tft.quaternion_about_axis(dangle, daxis)), t0, t1))
    
    # detect stationarity
    if tft.vector_norm(v_v) < 1e-1:
        stationary = True
        log.log('lo stationary! @ {}'.format(t0))
    else:
        stationary = False

    if kf.get_no_previous_states() < kf.STATE_BUFFER_LENGTH:
        log.log('lo frame discarded.')
        return

    if lo_rate_adjust % 8 == 0:
        lo_rate_adjust = 1
    else:
        lo_rate_adjust = lo_rate_adjust + 1
        return

    # velocity correction
    # def meas_fun(ns):
    #     R = tft.quaternion_matrix(ns.q())[0:3,0:3]
    #     return np.matmul(R.T, ns.v())
    # def constraints(dx):
    #     if not is_stationary() and np.linalg.norm(dx[0:3]) > 0.5:
    #         dx[0:3] = np.zeros(3)
    #         dx[6:9] = np.zeros(3)
    #     return dx
    # v_var = 1e-2
    # log.log('lo abs correction')
    # kf.correct_absolute(meas_fun, v_v, np.diag([v_var, v_var, v_var]), t1, constraints=constraints, measurement_name='visualodom_v')

    # relative rotation correction
    def meas_fun(ns1, ns0):
        qd_s = tft.quaternion_multiply(tft.quaternion_conjugate(ns0.q()),ns1.q())
        ds_angle, ds_axis = quaternion_to_angle_axis(qd_s)
        return ds_angle * ds_axis
    def hx_fun(ns1, ns0):
        qd_s = tft.quaternion_multiply(tft.quaternion_conjugate(ns0.q()),ns1.q())
        du_dq0 = np.dot(np.dot(jacobian_of_axisangle_wrt_q(qd_s), quaternion_right_multmat(ns1.q())), jacobian_of_qinv_wrt_q())
        du_dq1 = np.dot(jacobian_of_axisangle_wrt_q(qd_s), quaternion_left_multmat(tft.quaternion_conjugate(ns0.q())))
        Hx0 = np.concatenate([np.zeros((3,6)), du_dq0, np.zeros((3,6))], axis=1)
        Hx1 = np.concatenate([np.zeros((3,6)), du_dq1, np.zeros((3,6))], axis=1)
        return Hx1, Hx0
    kf.correct_relative(meas_fun, dtheta, np.ones(3)*1e-2, t1, t0, hx_fun=hx_fun, measurement_name='visualodom_q')


vo_rate_adjust = 1
def visualodom_callback(data):
    global stationary, vo_rate_adjust
    t1 = data.header.stamp.to_sec()
    t0 = data.twist.twist.linear.z
    
    # check for validity
    if t0 < 0:
        return

    # delta-position and quaternions in camera-init frame
    dp_ci = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
    ciqc1 = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    ciqc0 = np.array([data.twist.twist.angular.x, data.twist.twist.angular.y, data.twist.twist.angular.z, data.twist.twist.linear.x])
    ciRc0 = tft.quaternion_matrix(ciqc0)[0:3,0:3]
    ciRc1 = tft.quaternion_matrix(ciqc1)[0:3,0:3]

    # obtain vehicle to camera transform
    try:
        trans = tf2_buffer.lookup_transform(config['tf_frame_state'], config['tf_frame_camera'], rospy.Time(0))
    except:
        log.log('visualodom_callback: transformation error.')
        return
    vqc = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
    vRc = tft.quaternion_matrix(vqc)[0:3,0:3]

    v_ci = dp_ci / (t1-t0) # velocity in camera init frame
    v_c0 = np.matmul(ciRc0.T, v_ci) # velocity in old camera frame
    v_v = np.matmul(vRc, v_c0) # velocity in vehicle frame
    # ZUPT conditions
    v_v[0] = 0
    v_v[2] = 0

    # obtain velocity from ground truth
    # r1 = np.array([kd.groundtruth.interp_r(t1), kd.groundtruth.interp_p(t1), kd.groundtruth.interp_h(t1)])
    # q1 = tft.quaternion_from_euler(r1[0], r1[1], r1[2])
    # p1 = np.array([kd.groundtruth.interp_x(t1), kd.groundtruth.interp_y(t1), kd.groundtruth.interp_z(t1)])
    # p0 = np.array([kd.groundtruth.interp_x(t0), kd.groundtruth.interp_y(t0), kd.groundtruth.interp_z(t0)])
    # dp = p1-p0
    # q_mat = tft.quaternion_matrix(q1)[0:3, 0:3]
    # dp_v = np.dot(q_mat.T, dp)
    # v_v = dp_v / (t1-t0)

    
    # obtain relative rotation using visual odometry output
    c0qc1 = tft.quaternion_multiply(tft.quaternion_conjugate(ciqc0), ciqc1)
    v0qv1 = tft.quaternion_multiply(vqc, tft.quaternion_multiply(c0qc1, tft.quaternion_conjugate(vqc)))
    dangle, daxis = quaternion_to_angle_axis(v0qv1)
    dtheta = dangle * daxis # axis angle difference of c1-c0

    # obtain relative rotation using ground truth
    # r1 = np.array([kd.groundtruth.interp_r(t1), kd.groundtruth.interp_p(t1), kd.groundtruth.interp_h(t1)])
    # q1 = tft.quaternion_from_euler(r1[0], r1[1], r1[2])
    # R1 = tft.quaternion_matrix(q1)[0:3,0:3]
    # r0 = np.array([kd.groundtruth.interp_r(t0), kd.groundtruth.interp_p(t0), kd.groundtruth.interp_h(t0)])
    # q0 = tft.quaternion_from_euler(r0[0], r0[1], r0[2])
    # qd = tft.quaternion_multiply(tft.quaternion_conjugate(q0),q1)
    # dangle, daxis = quaternion_to_angle_axis(qd)
    # dtheta = dangle * daxis
        
    # detect stationarity
    if tft.vector_norm(v_v) < 1e-1:
        stationary = True
        log.log('vo stationary! @ {}'.format(t0))
    else:
        stationary = False

    if vo_rate_adjust % 5 == 0:
        vo_rate_adjust = 1
    else:
        vo_rate_adjust = vo_rate_adjust + 1
        return

    if kf.get_no_previous_states() < kf.STATE_BUFFER_LENGTH:
        log.log('vo frame discarded.')
        return

    # velocity correction
    log.log('vo abs correction')
    def meas_fun(ns):
        R = tft.quaternion_matrix(ns.q())[0:3,0:3]
        return np.matmul(R.T, ns.v())
    def constraints(dx):
        if not is_stationary() and np.linalg.norm(dx[0:3]) > 0.5:
            dx[0:3] = np.zeros(3)
            dx[6:9] = np.zeros(3)
        return dx
    v_var = 1e-6
    kf.correct_absolute(meas_fun, v_v * 0.98, np.diag([v_var, v_var, v_var]), t1, constraints=constraints, measurement_name='visualodom_v')

    # log.log("angle:{}, t0:{}, t1:{}".format(tft.euler_from_quaternion(tft.quaternion_about_axis(dangle, daxis)), t0, t1))

    # kf.print_states('before rel')

    # relative rotation correction
    # if not is_stationary():
    #     def meas_fun(ns1, ns0):
    #         qd_s = tft.quaternion_multiply(tft.quaternion_conjugate(ns0.q()),ns1.q())
    #         ds_angle, ds_axis = quaternion_to_angle_axis(qd_s)
    #         return ds_angle * ds_axis
    #     def hx_fun(ns1, ns0):
    #         qd_s = tft.quaternion_multiply(tft.quaternion_conjugate(ns0.q()),ns1.q())
    #         du_dq0 = np.dot(np.dot(jacobian_of_axisangle_wrt_q(qd_s), quaternion_right_multmat(ns1.q())), jacobian_of_qinv_wrt_q())
    #         du_dq1 = np.dot(jacobian_of_axisangle_wrt_q(qd_s), quaternion_left_multmat(tft.quaternion_conjugate(ns0.q())))
    #         Hx0 = np.concatenate([np.zeros((3,6)), du_dq0, np.zeros((3,6))], axis=1)
    #         Hx1 = np.concatenate([np.zeros((3,6)), du_dq1, np.zeros((3,6))], axis=1)
    #         return Hx1, Hx0
    #     kf.correct_relative(meas_fun, dtheta, np.eye(3)*1e-3, t1, t0, hx_fun=hx_fun, measurement_name='visualodom_q')

    # kf.print_states('after rel')

    # def meas_fun(ns1, ns0):
    #     r1 = tft.quaternion_matrix(ns1.q())[0:3,0:3]
    #     r0 = tft.quaternion_matrix(ns0.q())[0:3,0:3]
    #     # dp = np.matmul(r1.T, ns1.p())-np.matmul(r0.T,ns0.p())
    #     dp = ns1.p() - ns0.p()
    #     return dp
    # if stationary:
    #     kf.correct_relative(meas_fun, np.array((0.1,0,0)), np.eye(3)*1e-4, t1, t0, measurement_name='visualodom_q')
    

def publish_static_transforms(static_broadcaster):
    # world to map static transformation
    world2map_static_tf = TransformStamped()
    world2map_static_tf.header.stamp = rospy.Time.now()
    world2map_static_tf.header.frame_id = config['tf_frame_world']
    world2map_static_tf.child_frame_id = config['tf_frame_map']
    world2map_static_tf.transform.translation.x = 0
    world2map_static_tf.transform.translation.y = 0
    world2map_static_tf.transform.translation.z = 0
    world2map_static_tf.transform.rotation.w = 1
    world2map_static_tf.transform.rotation.x = 0
    world2map_static_tf.transform.rotation.y = 0
    world2map_static_tf.transform.rotation.z = 0

    # map to odom static transformation
    map2odom_static_tf = TransformStamped()
    map2odom_static_tf.header.stamp = rospy.Time.now()
    map2odom_static_tf.header.frame_id = config['tf_frame_map']
    map2odom_static_tf.child_frame_id = config['tf_frame_odom']
    map2odom_static_tf.transform.translation.x = 0
    map2odom_static_tf.transform.translation.y = 0
    map2odom_static_tf.transform.translation.z = 0
    map2odom_static_tf.transform.rotation.w = 1
    map2odom_static_tf.transform.rotation.x = 0
    map2odom_static_tf.transform.rotation.y = 0
    map2odom_static_tf.transform.rotation.z = 0

    static_broadcaster.sendTransform([world2map_static_tf, map2odom_static_tf])


if __name__ == '__main__':
    config = get_config_dict()['general']
    log = Log(config['locator_node_name'])

    rospy.init_node(config['locator_node_name'], anonymous=True)
    log.log('Node initialized.')

    rospy.Subscriber(config['processed_gnss_topic'], Odometry, gnss_callback, queue_size=1)
    rospy.Subscriber(config['processed_altitude_topic'], Odometry, alt_callback, queue_size=1)
    rospy.Subscriber(config['processed_imu_topic'], Imu, imu_callback, queue_size=1)
    rospy.Subscriber(config['processed_magneto_topic'], MagneticField, mag_callback, queue_size=1)
    rospy.Subscriber(config['processed_laserodom_topic'], Odometry, laserodom_callback, queue_size=1)
    rospy.Subscriber(config['processed_visualodom_topic'], Odometry, visualodom_callback, queue_size=1)

    state_pub = rospy.Publisher(config['state_topic'], State, queue_size=1)

    tf2_broadcaster = tf2.TransformBroadcaster()
    tf2_static_broadcaster = tf2.StaticTransformBroadcaster()

    tf2_buffer = tf2.Buffer()
    tf2_listener = tf2.TransformListener(tf2_buffer)

    publish_static_transforms(tf2_static_broadcaster)

    # get initial values for unobserved biases and velocity
    var_imu_linear_acceleration_bias  = np.array(rospy.get_param('/kalani/init/var_imu_linear_acceleration_bias') )
    var_imu_angular_velocity_bias     = np.array(rospy.get_param('/kalani/init/var_imu_angular_velocity_bias')    )
    init_velocity                     = np.array(rospy.get_param('/kalani/init/init_velocity')                    )
    init_var_velocity                 = np.array(rospy.get_param('/kalani/init/init_var_velocity')                )
    init_imu_linear_acceleration_bias = np.array(rospy.get_param('/kalani/init/init_imu_linear_acceleration_bias'))
    init_imu_angular_velocity_bias    = np.array(rospy.get_param('/kalani/init/init_imu_angular_velocity_bias')   )

    # Kalman filter definition
    def motion_model(ts, mmi, pn, dt):
        R = tft.quaternion_matrix(ts.q())[0:3, 0:3]  # body frame to enu frame transform matrix (from state quaternion)
        g = np.array([0, 0, config['gravity']])  # gravitational acceleration in ENU frame
        angle, axis = axisangle_to_angle_axis((mmi.w() + pn.w() - ts.wb())*dt)
        return np.concatenate([
            ts.p() + ts.v() * dt + 0.5 * (np.matmul(R, mmi.a() + pn.a() - ts.ab()) + g) * dt**2,
            ts.v() + (np.matmul(R, mmi.a() + pn.a() - ts.ab()) + g) * dt,
            tft.quaternion_multiply(ts.q(), tft.quaternion_about_axis(angle, axis)),
            ts.ab() + pn.ab(),
            ts.wb() + pn.wb(),
        ])

    def combination(ns, es):
        angle, axis = axisangle_to_angle_axis(es.q())
        try:
            return np.concatenate([
                ns.p() + es.p(),
                ns.v() + es.v(),
                tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), ns.q()),
                ns.ab() + es.ab(),
                ns.wb() + es.wb(),
            ])
        except Exception as e:
            log.log(e, angle, axis, es.q())

    def difference(ns1, ns0):
        angle, axis = quaternion_to_angle_axis(tft.quaternion_multiply(ns1.q(), tft.quaternion_conjugate(ns0.q())))
        return np.concatenate([
            ns1.p() - ns0.p(),
            ns1.v() - ns0.v(),
            angle * axis,
            ns1.ab() - ns0.ab(),
            ns1.wb() - ns0.wb(),
        ])

    def fx(ns, mmi, dt):
        R = tft.quaternion_matrix(ns.q())[0:3, 0:3]
        Fx = np.eye(15)
        Fx[0:3, 3:6] = np.eye(3) * dt
        Fx[3:6, 6:9] = -skew_symmetric(np.matmul(R, (mmi.a()-ns.ab()))) * dt
        Fx[3:6, 9:12] = -R * dt
        Fx[6:9, 12:15] = -R * dt
        return Fx

    def fi(ns, mmi, dt):
        Fi = np.concatenate([np.zeros((3, 12)), np.eye(12)], axis=0)
        return Fi

    def ts(ns):
        q = ns.q()
        qr = np.array([
            [ q[3],  q[2], -q[1], q[0]],
            [-q[2],  q[3],  q[0], q[1]],
            [ q[1], -q[0],  q[3], q[2]],
            [-q[0], -q[1], -q[2], q[3]]
        ])
        qt = 0.5 * np.concatenate([np.eye(3), np.zeros((1,3))])
        Q_dtheta = np.matmul(qr, qt)
        X = np.zeros([16, 15])
        X[0:6, 0:6] = np.eye(6)
        X[6:10, 6:9] = Q_dtheta
        X[10:16, 9:15] = np.eye(6)
        return X

    # ns_template, es_template, pn_template, mmi_template, motion_model(ts, mmi, pn, dt),
    # combination(ns, es), difference(ns1, ns0)
    kf = KalmanFilter(
        # nominal state template
        [
            ['p', 3],
            ['v', 3],
            ['q', 4],
            ['ab', 3],
            ['wb', 3],
        ],
        # error state template
        [
            ['p', 3],
            ['v', 3],
            ['q', 3],
            ['ab', 3],
            ['wb', 3],
        ],
        # process noise template
        [
            ['a', 3],
            ['w', 3],
            ['ab', 3],
            ['wb', 3],
        ],
        # motion model input template
        [
            ['a', 3],
            ['w', 3],
        ],
        motion_model,
        combination,
        difference,
        ts_fun=ts,
        fx_fun=fx,
        fi_fun=fi
    )

    kd = KAISTData()
    kd.load_data(groundtruth=True)

    log.log('Node ready.')

    rospy.spin()
