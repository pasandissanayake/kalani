#!/usr/bin/env python

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

aw_var = 0.001
ww_var = 0.001

am_var = 0.00001
wm_var = 0.00001

g = np.array([0, 0, -9.8])

# kf = Kalman_Filter_V1(g, aw_var, ww_var)
kf = Kalman_Filter_V1(g, aw_var, ww_var)

# Latest acceleration measured by the IMU, to be used in estimating orientation in mag_callback()
latest_acceleration = np.zeros(3)

# Laser localization attributes
camera_init_initialized = False
last_ld_transform = None
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
    # transform = tf2_buffer.lookup_transform('base_link', 'left_velodyne', data.header.stamp)
    # trans_cloud = do_transform_cloud(data, transform)
    pointcloud_pub.publish(data)

    try:
        o2l = tf2_buffer.lookup_transform('odom', 'aft_mapped', rospy.Time(0))
        l2o = tf2_buffer.lookup_transform('aft_mapped', 'odom', rospy.Time(0))
        # l2o = tf2_buffer.lookup_transform('odom', 'laser_data', rospy.Time(0))
        # o2l = tf2_buffer.lookup_transform('laser_data', 'odom', rospy.Time(0))
        time = o2l.header.stamp.to_sec()
        state = kf.get_state_as_numpy()

        global last_ld_transform, count_from_last_ld

        if last_ld_transform is None:
            if kf.is_initialized():
                last_ld_transform = [time, l2o, state]
        else:
            l1Ro = get_transform_matrix_from_ts(last_ld_transform[1])
            oRl2 = get_transform_matrix_from_ts(o2l)
            l1Rl2 = np.matmul(l1Ro, oRl2)

            r = np.array([[-0.78021039, 0.54694543, 0.30351678, 0],
                          [-0.62551718, -0.6822075, -0.37857784, 0],
                          [0., -0.48522533, 0.87438915, 0],
                          [0, 0, 0, 1]])
            l1Rl2 = np.matmul(r, l1Rl2)

            # print 'rel_trans:', l1Rl2[0:3, 3]

            prev_state = last_ld_transform[2]
            prev_rot = tft.quaternion_matrix([prev_state[7], prev_state[8], prev_state[9], prev_state[6]])
            prev_r = np.zeros((4,4))
            prev_r[0:3, 0:3] = prev_rot[0:3, 0:3]
            prev_r[0:3, 3] = prev_state[0:3]
            prev_r[3, 3] = 1
            l1Rl2 = np.matmul(prev_r, l1Rl2)

            translation = l1Rl2[0:3, 3]

            dt = time - last_ld_transform[0]
            count_from_last_ld = count_from_last_ld + 1
            if count_from_last_ld>1:
                count_from_last_ld = 0
                last_ld_transform = [time, l2o, state]
            if dt != 0:
                velo = translation / dt
                # print 'trans:', translation, 'original:', state[0:3]
                # def meas_func(state):
                #     r = tft.quaternion_matrix([state[7], state[8], state[9], state[6]])[0:3, 0:3]
                #     return velo - np.matmul(r.T, state[3:6])
                #
                # def hx_func(state):
                #     def f(x):
                #         vx, vy, vz = x[3:6]
                #         qw, qx, qy, qz = x[6:10]
                #
                #         return anp.array([
                #             (1 - 2 * (qy ** 2 + qz ** 2)) * vx + 2 * (qx * qy + qz * qw) * vy + 2 * (
                #                         qx * qz - qy * qw) * vz,
                #             2 * (qx * qy - qw * qz) * vx + (1 - 2 * (qx ** 2 + qz ** 2)) * vy + 2 * (
                #                         qy * qz + qx * qw) * vz,
                #             2 * (qx * qz + qy * qw) * vx + 2 * (qy * qz - qx * qw) * vy + (
                #                         1 - 2 * (qx ** 2 + qy ** 2)) * vz
                #         ])
                #
                #     jacob = jacobian(f)
                #     return np.array(jacob(state))

                # def meas_func(state):
                #     r = tft.quaternion_matrix([state[7], state[8], state[9], state[6]])[0:3, 0:3]
                #     return velo[1] - np.matmul(r.T, state[3:6])[1]
                #
                # def hx_func(state):
                #     def f(x):
                #         vx, vy, vz = x[3:6]
                #         qw, qx, qy, qz = x[6:10]
                #
                #         return anp.array([
                #
                #             2 * (qx * qy - qw * qz) * vx + (1 - 2 * (qx ** 2 + qz ** 2)) * vy + 2 * (
                #                         qy * qz + qx * qw) * vz
                #         ])
                #
                #     jacob = jacobian(f)
                #     return np.array(jacob(state))

                def meas_func(new_state):
                    print 'dif:', translation - state[0:3]
                    return translation - new_state[0:3]

                def hx_func(new_state):
                    hx = np.zeros((3,16))
                    hx[:, 0:3] = np.eye(3)
                    return hx

                V = np.diag([1, 1, 1])
                # V = np.diag([0.00001])
                # kf.correct(meas_func, hx_func, V, time, measurementname='laser')

    except Exception as e:
        print e.message


def laser_dt_callback(data):
    time = data.header.stamp.to_sec()
    try:
        lv2bl_msg = tf2_buffer.lookup_transform('base_link', 'left_velodyne', rospy.Time(0))
        lv2bl_mat = get_transform_matrix_from_ts(lv2bl_msg)
        dp = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z, 0])
        dp_bl = np.matmul(lv2bl_mat, dp)

        dr_quat = (data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        am0_R_am1 = tft.quaternion_matrix(dr_quat)[0:3, 0:3]
        ld_R_am = lv2bl_mat[0:3, 0:3]
        ld0_R_ld1 = np.matmul(np.matmul(ld_R_am, am0_R_am1), ld_R_am.T)
        dq_bl = quaternion_from_3x3_matrix(ld0_R_ld1)

        # for i in range(3):
        #     if abs(dp_bl[i]) < 0.1:
        #         dp_bl[i] = 0
        print 'time', time, 'dp_bl', dp_bl[0:3], 'yaw', tft.euler_from_quaternion(dq_bl)[2] * 180 / np.pi

        velo = dp_bl[0:3] / 0.1

        def meas_func(state):
            r = tft.quaternion_matrix([state[7], state[8], state[9], state[6]])[0:3, 0:3]
            return velo - np.matmul(r.T, state[3:6])

        def hx_func(state):
            def f(x):
                vx, vy, vz = x[3:6]
                qw, qx, qy, qz = x[6:10]

                return anp.array([
                    (1 - 2 * (qy ** 2 + qz ** 2)) * vx + 2 * (qx * qy + qz * qw) * vy + 2 * (
                            qx * qz - qy * qw) * vz,
                    2 * (qx * qy - qw * qz) * vx + (1 - 2 * (qx ** 2 + qz ** 2)) * vy + 2 * (
                            qy * qz + qx * qw) * vz,
                    2 * (qx * qz + qy * qw) * vx + 2 * (qy * qz - qx * qw) * vy + (
                            1 - 2 * (qx ** 2 + qy ** 2)) * vz
                ])

            jacob = jacobian(f)
            return np.array(jacob(state))

        V = np.diag([1, 1, 1]) * 1
        # kf.correct(meas_func, hx_func, V, time-0.1, measurementname='laser')
    except Exception as e:
        print e.message


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

    # qr = np.array([[-0.486485, 0.711672, -0.506809, 0],
    #                [-0.514066, -0.702201, -0.492595, 0],
    #                [-0.706447, 0.0208933, 0.707457, 0],
    #                [        0,         0,        0, 1]])
    #
    # q = tft.quaternion_from_matrix(qr)

    q = tft.quaternion_from_euler(0.029524002802542637, 0.7844662631054065, -2.3286358259713964, axes='sxyz')

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
