#!/usr/bin/env python2

import rospy
import tf.transformations as tft
import tf
import tf2_py as tf2
import tf2_ros
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from kalani_v1.msg import State
import pcl_ros
import numpy as np
import scipy
from scipy.optimize import leastsq
from constants import Constants
from filter.kalman_filter_v1 import Kalman_Filter_V1
from datasetutils.kaist_data_conversions import KAISTDataConversions
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_geometry_msgs
import autograd.numpy as anp
from autograd import jacobian

loam_var = 0.001 * np.ones(3)
mag_orientation_var = 0.01 * np.ones(3)

aw_var = 0.00001
ww_var = 0.00001

am_var = 0.000001
wm_var = 0.000001

g = np.array([0, 0, -9.8])

# kf = Kalman_Filter_V1(g, aw_var, ww_var)
kf = Kalman_Filter_V1(g, aw_var, ww_var)

# Latest acceleration measured by the IMU, to be used in estimating orientation in mag_callback()
latest_acceleration = np.zeros(3)

# Laser localization attributes
camera_init_initialized = False
ld_last_pose = None
count_from_last_ld = 0


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

    transform = TransformStamped()
    transform.header.stamp = msg.header.stamp
    transform.header.frame_id = 'odom'
    transform.child_frame_id = 'base_link'
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z  = state[0:3]
    transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z = state[6:10]
    tf2_broadcaster.sendTransform(transform)


def publish_gnss(pub, time, fix):
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.position.x, msg.position.y, msg.position.z = list(fix)
    pub.publish(msg)

    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(time)
    transform.header.frame_id = 'odom'
    transform.child_frame_id = 'gnss_data'
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = list(fix)
    transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z = (1, 0, 0, 0)
    tf2_broadcaster.sendTransform(transform)


def publish_magnetic(pub, ori, time):
    msg = State()
    msg.header.stamp = rospy.Time.from_sec(time)
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = ori
    [msg.euler.x, msg.euler.y, msg.euler.z] = tft.euler_from_quaternion(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], axes='sxyz')
    pub.publish(msg)

    transform = TransformStamped()
    transform.header.stamp = rospy.Time.from_sec(time)
    transform.header.frame_id = 'odom'
    transform.child_frame_id = 'mag_data'
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (10, 10, 0)
    transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z = ori
    tf2_broadcaster.sendTransform(transform)


def publish_mag_orientation(time, q):
    euler = tft.euler_from_quaternion([q[1],q[2],q[3],q[0]],axes='sxyz')
    pub = rospy.Publisher('converted_data/_mag_ori', State, queue_size=1)
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
    time = data.header.stamp.to_sec()
    lat = data.latitude
    long = data.longitude
    alt = data.altitude
    covariance = data.position_covariance
    gnss_array = np.concatenate([[time*1e9,lat,long,alt],covariance])
    gnss = KAISTDataConversions.gnss_numpy_to_converted(gnss_array)
    fix = np.array([gnss.x,gnss.y,gnss.z])

    if kf.is_initialized():
        Hx = np.zeros([3, 16])
        Hx[:, 0:3] = np.eye(3)
        V = np.diag([covariance[0],covariance[4],covariance[8] * 10])

        def hx_func(state):
            return Hx

        def meas_func(state):
            return fix - state[0:3]

        kf.correct(meas_func, hx_func, V, time, measurementname='gnss')

        # Hx = np.zeros([2, 16])
        # Hx[:, 0:2] = np.eye(2)
        # V = np.diag([covariance[0], covariance[4]])
        # kf.correct(fix[0:2], Hx, V, time, measurementname='gnss')

        publish_state(state_pub)
        publish_gnss(converted_gnss_pub, time, fix)

    else:
        p = fix
        cov_p = np.array([covariance[0],covariance[4],covariance[8]])

        v = np.zeros(3)
        cov_v = [0,0,0]

        ab = np.array([0.0, 0.0, 0.0])
        cov_ab = aw_var * np.ones(3)

        wb = np.zeros(3)
        cov_wb = ww_var * np.ones(3)

        t = time

        kf.initialize_state(p=p,cov_p=cov_p,v=v,cov_v=cov_v,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,time=t)

        # odom to base_link_init static transformation
        state = kf.get_state_as_numpy()
        odom2bli_tf = TransformStamped()
        odom2bli_tf.header.stamp = rospy.Time.from_sec(time)
        odom2bli_tf.header.frame_id = 'odom'
        odom2bli_tf.child_frame_id = 'base_link_init'
        odom2bli_tf.transform.translation.x = state[0]
        odom2bli_tf.transform.translation.y = state[1]
        odom2bli_tf.transform.translation.z = state[2]

        odom2bli_tf.transform.rotation.w = state[6]
        odom2bli_tf.transform.rotation.x = state[7]
        odom2bli_tf.transform.rotation.y = state[8]
        odom2bli_tf.transform.rotation.z = state[9]

        print 'quaternion_init:', state[6:10]

        tf2_static_broadcaster.sendTransform(odom2bli_tf)


def imu_callback(data):
    am = KAISTDataConversions.vector_nwu_to_enu(np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))
    global latest_acceleration
    latest_acceleration = am

    am = np.array([am[0],am[1],am[2]])

    wm = KAISTDataConversions.vector_nwu_to_enu(np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z]))

    time = data.header.stamp.to_sec()

    if kf.is_initialized():
        kf.predict(am, am_var, wm, wm_var, time, inputname='imu')

        # zero velocity update

        def hx_func(state):
            vx, vy, vz = state[3:6]
            qw, qx, qy, qz = state[6:10]

            Hx = np.array([
                [0, 0, 0, 1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw),
                 2 * (qz * vy - qy * vz), 2 * (qy * vy + qz * vz), 2 * (qx * vy - qw * vz - 2 * qy * vx),
                 2 * (qw * vy + qx * vz - 2 * qz * vz), 0, 0, 0, 0, 0, 0], # problem in first element of this line
                [0, 0, 0, 2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx ** 2 + qy ** 2),
                 2 * (qy * vx - qx * vy), 2 * (qz * vx - qw * vy - 2 * qx * vz), 2 * (qw * vx + qz * vy - 2 * qy * vz),
                 2 * (qx * vx + qy * vy), 0, 0, 0, 0, 0, 0]
            ])
            return Hx

        def meas_func(state):
            vx, vy, vz = state[3:6]
            qw, qx, qy, qz = state[6:10]
            y_expect = np.array([
                (1 - 2 * (qy**2 + qz**2)) * vx + 2 * (qx*qy + qz*qw) * vy + 2 * (qx*qz - qy*qw) * vz ,
                2 * (qx*qz + qy*qw) * vx + 2 * (qy*qz - qx*qw) * vy + (1 - 2*(qx**2 + qy**2)) * vz
            ])

            return np.zeros(2) - y_expect

        V = np.diag([10000, 1])

        # kf.correct(meas_func, hx_func, V, time, measurementname='zupt')

        publish_state(state_pub)


def mag_callback(data):
    mm = KAISTDataConversions.vector_nwu_to_enu(np.array([data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z]))
    time = data.header.stamp.to_sec()

    ori = get_orientation_from_magnetic_field(mm, latest_acceleration)

    if kf.is_initialized():

        def hx_func(state):
            Hx = np.zeros([4, 16])
            Hx[:, 6:10] = np.eye(4)
            return Hx

        def meas_func(state):
            return ori - state[6:10]

        V = np.diag(np.ones(4) * 0.1)
        # kf.correct(meas_func, hx_func, V, time, measurementname='magnetometer')
        publish_state(state_pub)
        publish_magnetic(converted_mag_pub, ori, time)

    else:
        v = np.zeros(3)
        cov_v = [0, 0, 0]

        q = ori
        cov_q = mag_orientation_var

        ab = np.zeros(3)
        cov_ab = aw_var * np.ones(3)

        wb = np.zeros(3)
        cov_wb = ww_var * np.ones(3)

        kf.initialize_state(v=v,cov_v=cov_v,q=q,cov_q=cov_q,ab=ab,cov_ab=cov_ab,wb=wb,cov_wb=cov_wb,g=g,time=time)

        if kf.is_initialized():
            log('State initialized.')


def get_transform_matrix_from_ts(ts_msg):
    q = np.array([ts_msg.transform.rotation.x, ts_msg.transform.rotation.y, ts_msg.transform.rotation.z, ts_msg.transform.rotation.w])
    t = np.array([ts_msg.transform.translation.x, ts_msg.transform.translation.y, ts_msg.transform.translation.z])
    R = tft.quaternion_matrix(q)
    R[0:3, 3] = t.T
    return R


def quaternion_from_3x3_matrix(matrix):
    mat = np.zeros((4,4))
    mat[0:3, 0:3] = matrix
    mat[3, 3] = 1
    return tft.quaternion_from_matrix(mat)


def laser_callback(data):
    pointcloud_pub.publish(data)


def laser_dt_callback(data):
    time = data.header.stamp.to_sec()

    lv2bl_msg = tf2_buffer.lookup_transform('base_link', 'left_velodyne', rospy.Time(0))
    lv2bl_mat = get_transform_matrix_from_ts(lv2bl_msg)

    dp_velodyne = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z, 0])
    dp_laser = np.matmul(lv2bl_mat, dp_velodyne)[0:3]

    dq_velodyne = (data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
    lv1_R_lv2 = tft.quaternion_matrix(dq_velodyne)[0:3, 0:3]
    bl_R_lv = lv2bl_mat[0:3, 0:3]
    bl1_R_bl2 = np.matmul(np.matmul(bl_R_lv, lv1_R_lv2), bl_R_lv.T)
    dq_laser = quaternion_from_3x3_matrix(bl1_R_bl2)

    for i in range(3):
        if abs(dp_laser[i]) < 0.1:
            dp_laser[i] = 0

    def meas_func(state):
        rot_mat = tft.quaternion_matrix((state[7], state[8], state[9], state[6]))[0:3, 0:3].T
        dp_state = np.matmul(rot_mat,state[16:19] - state[0:3])

        q_state_1 = (state[7], state[8], state[9], state[6])
        q_state_2 = (state[23], state[24], state[25], state[22])
        dq_state = tft.quaternion_multiply(q_state_2, tft.quaternion_conjugate(q_state_1))

        eq = tft.quaternion_multiply(dq_laser, tft.quaternion_conjugate(dq_state))
        eq = np.concatenate([[eq[3]], eq[0:3]])

        print 'state dp', dp_state
        print 'laser dp', dp_laser
        print 'state dq', dq_state
        print 'laser dq', dq_laser
        print 'error dp', dp_laser - dp_state
        print 'error dq', eq
        return np.concatenate([dp_laser - dp_state, eq])

    def hx_func(state):
        qw, qx, qy, qz = state[6:10]
        rw, rx, ry, rz = state[22:26]
        vx, vy, vz = state[16:19] - state[0:3]
        Hx = np.array([
            [-(1-2*qy**2-2*qz**2), -(2*qx*qy+2*qz*qw), -(2*qx*qz-2*qy*qw), 0,0,0, 2*qz*vy-2*qy*vz, 2*qy*vy+2*qz*vz, -4*qy*vx+2*qx*vy-2*qw*vz, -4*qz*vx+2*qw*vy+2*qx*vz, 0,0,0,0,0,0,
             1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy + 2 * qz * qw, 2 * qx * qz - 2 * qy * qw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-(2*qx*qy-2*qz*qw), -(1-2*qx**2-2*qz**2), -(2*qy*qz+2*qx*qw), 0,0,0, -2*qz*vx+2*qx*vz, 2*qy*vx-4*qx*vy+2*qw*vz, 2*qx*vx+2*qz*vz, -2*qw*vx-4*qz*vy+2*qy*vz, 0,0,0,0,0,0,
             2 * qx * qy - 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz + 2 * qx * qw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-(2*qx*qz+2*qy*qw), -(2*qy*qz-2*qx*qw), -(1-2*qx**2-2*qy**2), 0,0,0, 2*qy*vx-2*qx*vy, 2*qz*vx-2*qw*vy-4*qx*vz, 2*qw*vx+2*qz*vy-4*qy*vz, 2*qx*vx+2*qy*vy, 0,0,0,0,0,0,
             2 * qx * qz + 2 * qy * qw, 2 * qy * qz - 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0,0,0,0,0,0, qw, qx,  qy,  qz, 0,0,0,0,0,0, 0,0,0,0,0,0,  rw, rx, ry,  rz, 0,0,0,0,0,0],
            [0,0,0,0,0,0, qx, -qw, -qz, qy, 0,0,0,0,0,0, 0,0,0,0,0,0, -rx, rw, rz, -ry, 0,0,0,0,0,0],
            [0,0,0,0,0,0, qy, qz, -qw, -qx, 0,0,0,0,0,0, 0,0,0,0,0,0, -ry, -rz, rw, rx, 0,0,0,0,0,0],
            [0,0,0,0,0,0, qz, -qy, qx, -qw, 0,0,0,0,0,0, 0,0,0,0,0,0, -rz, ry, -rx, rw, 0,0,0,0,0,0],
        ])
        # print 'jacob_state', np.matmul(Hx, state)
        return Hx

    V = np.diag([1, 1, 1, 1, 1, 1, 1]) * 0.01
    kf.correct_relative(meas_func, hx_func, V, time-0.1, time, measurementname='laser')


if __name__ == '__main__':
    rospy.init_node(Constants.LOCATOR_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber('/gps/fix', NavSatFix, gnss_callback, queue_size=1)
    rospy.Subscriber('/imu/data_raw', Imu, imu_callback, queue_size=1)
    rospy.Subscriber('/imu/mag', MagneticField, mag_callback, queue_size=1)
    rospy.Subscriber('/ns2/velodyne_points', PointCloud2, laser_callback, queue_size=1)
    rospy.Subscriber('/laser_odom_dt', Odometry, laser_dt_callback, queue_size=1)

    state_pub = rospy.Publisher(Constants.STATE_TOPIC, State, queue_size=1)
    converted_gnss_pub = rospy.Publisher('/converted_data/gnss', State, queue_size=1)
    converted_mag_pub = rospy.Publisher('/converted_data/magnetic', State, queue_size=1)
    pointcloud_pub = rospy.Publisher('/raw_data/velodyne_points', PointCloud2, queue_size=1)

    tf2_broadcaster = tf2_ros.TransformBroadcaster()
    tf2_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

    tf2_buffer = tf2_ros.Buffer()
    tf2_listener = tf2_ros.TransformListener(tf2_buffer)

    # world to map static transformation
    world2map_static_tf = TransformStamped()
    world2map_static_tf.header.stamp = rospy.Time.now()
    world2map_static_tf.header.frame_id = 'world'
    world2map_static_tf.child_frame_id = 'map'
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
    map2odom_static_tf.header.frame_id = 'map'
    map2odom_static_tf.child_frame_id = 'odom'
    map2odom_static_tf.transform.translation.x = 0
    map2odom_static_tf.transform.translation.y = 0
    map2odom_static_tf.transform.translation.z = 0
    map2odom_static_tf.transform.rotation.w = 1
    map2odom_static_tf.transform.rotation.x = 0
    map2odom_static_tf.transform.rotation.y = 0
    map2odom_static_tf.transform.rotation.z = 0

    # base_link to left_velodyne static transformation
    bl2lv_static_tf = TransformStamped()
    bl2lv_static_tf.header.stamp = rospy.Time.now()
    bl2lv_static_tf.header.frame_id = 'base_link'
    bl2lv_static_tf.child_frame_id = 'left_velodyne'
    bl2lv_static_tf.transform.translation.x = -0.397052
    bl2lv_static_tf.transform.translation.y = -0.440699
    bl2lv_static_tf.transform.translation.z = 1.90953

    qr = np.array([[-0.486485, 0.711672, -0.506809, 0],
                   [-0.514066, -0.702201, -0.492595, 0],
                   [-0.706447, 0.0208933, 0.707457, 0],
                   [        0,         0,        0, 1]])

    # q = tft.quaternion_from_euler(0.029524002802542637, 0.7844662631054065, -2.3286358259713964, axes='sxyz')

    # w_to_bli = tft.quaternion_matrix((0.00274975, 0.01072993, -0.90751944, 0.41986399))[0:3, 0:3]
    # cw_to_w = np.array([[0.97728753, 0.20838559, -0.03852966],
    #                     [-0.2118229, 0.95513289, -0.20700778],
    #                     [0.00633649, 0.21046759, 0.9775803]])
    # new_r = np.matmul(w_to_bli.T, cw_to_w)
    # new_r = np.matmul(new_r, w_to_bli)
    # bli_to_ci = np.array([[-0.486485, 0.711672, -0.506809],
    #                       [-0.514066, -0.702201, -0.492595],
    #                       [-0.706447, 0.0208933, 0.707457]])
    # new_r = np.matmul(new_r, bli_to_ci)
    #
    # qr = np.zeros((4, 4))
    # qr[0:3, 0:3] = new_r
    # qr[3, 3] = 1
    q = tft.quaternion_from_matrix(qr)

    bl2lv_static_tf.transform.rotation.w = q[3]
    bl2lv_static_tf.transform.rotation.x = q[0]
    bl2lv_static_tf.transform.rotation.y = q[1]
    bl2lv_static_tf.transform.rotation.z = q[2]

    # base_link_init to camera_init static transformation
    bli2ci_static_tf = TransformStamped()
    bli2ci_static_tf.header.stamp = rospy.Time.now()
    bli2ci_static_tf.header.frame_id = 'base_link_init'
    bli2ci_static_tf.child_frame_id = 'camera_init'
    bli2ci_static_tf.transform = bl2lv_static_tf.transform

    # aft_mapped to static transformation
    am2ld_static_tf = TransformStamped()
    am2ld_static_tf.header.stamp = rospy.Time.now()
    am2ld_static_tf.header.frame_id = 'aft_mapped'
    am2ld_static_tf.child_frame_id = 'laser_data'

    bl2lv_matrix = get_transform_matrix_from_ts(bl2lv_static_tf)
    bl2lv_inv_matrix = tft.inverse_matrix(bl2lv_matrix)

    am2ld_static_tf.transform.translation.x = bl2lv_inv_matrix[0, 3]
    am2ld_static_tf.transform.translation.y = bl2lv_inv_matrix[1, 3]
    am2ld_static_tf.transform.translation.z = bl2lv_inv_matrix[2, 3]
    q_am_ld = tft.quaternion_from_matrix(bl2lv_inv_matrix)
    am2ld_static_tf.transform.rotation.w = q_am_ld[3]
    am2ld_static_tf.transform.rotation.x = q_am_ld[0]
    am2ld_static_tf.transform.rotation.y = q_am_ld[1]
    am2ld_static_tf.transform.rotation.z = q_am_ld[2]

    tf2_static_broadcaster.sendTransform([bli2ci_static_tf, world2map_static_tf, map2odom_static_tf, bl2lv_static_tf, am2ld_static_tf])

    log('Locator ready.')
    rospy.spin()
