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
tprev_visualodom = None

# placeholders for message sequence numbers
seq_imu = -1
seq_gnss = -1
seq_altimeter = -1

# absolute stationarity condition (detected by odometers)
stationary = False

def is_stationary():
    # return stationary
    return False


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
    if data.header.seq > seq_gnss + 1:
        log.log("{} GNSS messages lost at {} s!".format(data.header.seq - seq_gnss -1, t))
    seq_gnss = data.header.seq

    # prepare pseudo-gnss fix
    cov = np.reshape(data.pose.covariance, (6,6))[0:2, 0:2]
    gnss_fix = np.array([kd.groundtruth.interp_x(t), kd.groundtruth.interp_y(t)]) + np.random.normal(0, np.sqrt(np.max(cov)), (2))
        
    if kf.is_initialized:
        def meas_fun(ns):
            return ns.p()[0:2]
        def hx_fun(ns):
            return np.concatenate([np.eye(2),np.zeros((2,14))], axis=1)
        if not is_stationary():
            kf.correct_absolute(meas_fun, gnss_fix, cov, t, hx_fun=hx_fun, measurement_name='gnss')
            pass
        publish_gnss(t, gnss_fix)

        # zero velocity update
        def meas_fun(ns):
            r = tft.quaternion_matrix(ns.q())[0:3, 0:3]
            r = r.T
            v = np.matmul(r, ns.v())
            return np.array([v[0], v[2]])
        # kf.correct_absolute(meas_fun, np.zeros(2), np.eye(2) * 1e-5, t, measurement_name='zupt')
    elif altitude is not None:
        cov_p = np.eye(3)
        cov_p[0:2, 0:2] = cov
        cov_p[2, 2] = var_altitude
        p = np.concatenate([gnss_fix, [altitude]])
        kf.initialize([
            ['p', p, cov_p, t],
            ['v', init_velocity, np.diag(init_var_velocity), t],
            ['ab', init_imu_linear_acceleration_bias, np.diag(var_imu_linear_acceleration_bias), t],
            ['wb', init_imu_angular_velocity_bias, np.diag(var_imu_angular_velocity_bias), t]
        ])


def alt_callback(data):
    global altitude, var_altitude, seq_altimeter
    t = data.header.stamp.to_sec()
    if data.header.seq > seq_altimeter + 1:
        log.log("{} Altimeter messages lost at {} s!".format(data.header.seq - seq_altimeter -1, t))
    seq_altimeter = data.header.seq

    # prepare pseudo-altitude measurement
    var_altitude = np.reshape(data.pose.covariance, (6, 6))[2, 2]
    altitude = kd.groundtruth.interp_z(t) + np.random.normal(0, np.sqrt(var_altitude))

    if kf.is_initialized:
        def meas_fun(ns):
            return np.array([ns.p()[2]])
        def hx_fun(ns):
            return np.concatenate([[0,0,1], np.zeros(13)]).reshape((1, 16))
        kf.correct_absolute(meas_fun, np.array([altitude]), var_altitude.reshape((1,1)), t, hx_fun=hx_fun, measurement_name='altitude')


def imu_callback(data):
    global linear_acceleration, angular_velocity, seq_imu
    t = data.header.stamp.to_sec()
    if data.header.seq > seq_imu + 1:
        log.log("{} IMU messages lost at {} s!".format(data.header.seq - seq_imu -1, t))
    seq_imu = data.header.seq
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
        inputs = np.concatenate([linear_acceleration, angular_velocity])
        kf.predict(inputs, cov, t, input_name='imu')
        publish_state()


def mag_callback(data):
    global magnetic_field, linear_acceleration
    t = data.header.stamp.to_sec()
    magnetic_field = np.array([data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z])
    cov = np.reshape(data.magnetic_field_covariance, (3,3))
    orientation = get_orientation_from_magnetic_field(magnetic_field, linear_acceleration)
    angle, axis = quaternion_to_angle_axis(orientation)
    meas_axisangle = angle * axis
    if kf.is_initialized:
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
            ['q', orientation, cov, t],
            ['ab', init_imu_linear_acceleration_bias, np.diag(var_imu_linear_acceleration_bias), t],
            ['wb', init_imu_angular_velocity_bias, np.diag(var_imu_angular_velocity_bias), t]
        ])


def laserodom_callback(data):
    pass

def visualodom_callback(data):
    global tprev_visualodom, stationary
    t = data.header.stamp.to_sec()
    if tprev_visualodom is None:
        tprev_visualodom = t
        return
    
    r1 = np.array([kd.groundtruth.interp_r(t+0.1), kd.groundtruth.interp_p(t+0.1), kd.groundtruth.interp_h(t+0.1)])
    q1 = tft.quaternion_from_euler(r1[0], r1[1], r1[2])
    R1 = tft.quaternion_matrix(q1)[0:3,0:3]
    p1 = np.array([kd.groundtruth.interp_x(t+0.1), kd.groundtruth.interp_y(t+0.1), kd.groundtruth.interp_z(t+0.1)])
    p0 = np.array([kd.groundtruth.interp_x(t), kd.groundtruth.interp_y(t), kd.groundtruth.interp_z(t)])
    r0 = np.array([kd.groundtruth.interp_r(t), kd.groundtruth.interp_p(t), kd.groundtruth.interp_h(t)])
    q0 = tft.quaternion_from_euler(r0[0], r0[1], r0[2])
    qd = tft.quaternion_multiply(tft.quaternion_conjugate(q0),q1)
    d_angle, d_axis = quaternion_to_angle_axis(qd)
    da = d_angle * d_axis
    
    vgt_var = 5e-2 / 0.1
    vgt_w = (p1 - p0) / (0.1) + np.random.normal(0, np.sqrt(vgt_var), 3)
    vgt_v = np.matmul(R1.T, vgt_w)
    vgt_v[0] = 0
    vgt_v[2] = 0
    log.log('vgt_v:', vgt_v)
    if tft.vector_norm(vgt_v) < 1e-2:
        stationary = True
    else:
        stationary = False
    def meas_fun(ns):
        R = tft.quaternion_matrix(ns.q())[0:3,0:3]
        return np.matmul(R.T, ns.v())
    kf.correct_absolute(meas_fun, vgt_v, np.diag([vgt_var * 1e-6, vgt_var, vgt_var * 1e-6 ]), t,measurement_name='visualodom')
    def meas_fun(ns1, ns0):
        qd_s = tft.quaternion_multiply(tft.quaternion_conjugate(ns0.q()),ns1.q())
        ds_angle, ds_axis = quaternion_to_angle_axis(qd_s)
        return ds_angle * ds_axis
    # kf.correct_relative(meas_fun, da, np.eye(3)*1e-5,t+0.1,t,measurement_name='visualodom_q')


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

    # Kalman filter formation
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
        return np.concatenate([
            ns.p() + es.p(),
            ns.v() + es.v(),
            tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), ns.q()),
            ns.ab() + es.ab(),
            ns.wb() + es.wb(),
        ])

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
