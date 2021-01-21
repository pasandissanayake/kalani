#!/usr/bin/env python2

import numpy as np
import numdifftools as numdiff

import rospy
import tf.transformations as tft
import tf2_ros as tf2

from kalani_v1.msg import State
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from kalman_filter import KalmanFilter
from utilities import *


# placeholders for most recent measurement values
gnss_fix = None
linear_acceleration = None
angular_acceleration = None
magnetic_field = None
altitude = None
var_altitude = None


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
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = list(fix)
    transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z = (1, 0, 0, 0)
    tf2_broadcaster.sendTransform(transform)


def publish_magnetic(timestamp, ori):
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(timestamp)
    transform.header.frame_id = config['tf_frame_world']
    transform.child_frame_id = config['tf_frame_magneto']
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = list(gnss_fix)
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = list(ori)
    tf2_broadcaster.sendTransform(transform)


def gnss_callback(data):
    global gnss_fix, altitude, var_altitude
    t = data.header.stamp.to_sec()
    gnss_fix = np.array([data.pose.pose.position.x, data.pose.pose.position.y, altitude])
    cov = np.reshape(data.pose.covariance, (6,6))[0:3, 0:3]
    cov[2, 2] = var_altitude
    if kf.is_initialized and altitude is not None:
        def meas_fun(ns):
            return ns.p()
        kf.correct_absolute(meas_fun, gnss_fix, cov, t, measurement_name='gnss')
        publish_gnss(t, gnss_fix)
    elif altitude is not None:
        kf.initialize([
            ['p', gnss_fix, cov, t],
            ['v', init_velocity, np.diag(init_var_velocity), t],
            ['ab', init_imu_linear_acceleration_bias, np.diag(var_imu_linear_acceleration_bias), t],
            ['wb', init_imu_angular_velocity_bias, np.diag(var_imu_angular_velocity_bias), t]
        ])


def alt_callback(data):
    global altitude, var_altitude
    altitude = data.pose.pose.position.z
    var_altitude = np.reshape(data.pose.covariance, (6, 6))[2, 2]


def imu_callback(data):
    global linear_acceleration, angular_velocity
    t = data.header.stamp.to_sec()
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
        kf.correct_absolute(meas_fun, meas_axisangle, cov, t, measurement_name='magnetometer')
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
            tft.quaternion_multiply(ns.q(), tft.quaternion_about_axis(angle, axis)),
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
        difference
    )

    log.log('Node ready.')

    rospy.spin()