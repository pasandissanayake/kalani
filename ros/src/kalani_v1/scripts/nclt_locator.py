#!/usr/bin/env python

import rospy
import tf.transformations as tft
import tf
import tf2_ros
import pcl_ros
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import PointCloud, PointField
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Header
from kalani_v1.msg import State

import numpy as np
import scipy
from scipy.optimize import leastsq
from constants import Constants
from filter.kalman_filter_v1 import Kalman_Filter_V1
from datasetutils.nclt_data_conversions import NCLTDataConversions
import numdifftools as nd


nclt_gnss_var = [25.0, 25.0, 100]
nclt_loam_var = 0.001 * np.ones(3)
nclt_mag_orientation_var = 0.1 * np.ones(3)

aw_var = 0.0001
ww_var = 0.0001

am_var = 0.01
wm_var = 0.001

g = np.array([0, 0, -9.8])

kf = Kalman_Filter_V1(g, aw_var, ww_var)

# Latest acceleration measured by the IMU, to be used in estimating orientation in mag_callback()
latest_acceleration = np.zeros(3)

# Previous measurement time for laser_dt_callback()
laser_dt_prev_time = -1


def log(message):
    rospy.loginfo(Constants.LOCATOR_NODE_NAME + ' := ' + str(message))


def tfq_2_csq(tf_q):
    return np.concatenate([[tf_q[3]], tf_q[0:3]])

def csq_2_tfq(cs_q):
    return np.concatenate([cs_q[1:4], [cs_q[0]]])

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

            # use all x, y and z
            def hx_func(state):
                Hx = np.zeros([3, 16])
                Hx[:, 0:3] = np.eye(3)
                return Hx
            def meas_func(state):
                return fix - state[0:3]
            V = np.diag(nclt_gnss_var)

            # # use x, y only even z is available
            # def hx_func(state):
            #     Hx = np.zeros([2, 16])
            #     Hx[:, 0:2] = np.eye(2)
            #     return Hx
            # def meas_func(state):
            #     return fix[0:2] - state[0:2]
            # V = np.diag(nclt_gnss_var[0:2])

            kf.correct(meas_func, hx_func, V, time, measurementname='gnss')
            publish_state(state_pub)
            publish_gnss(converted_gnss_pub, time, fix)

        else:
            fix = np.array([107.724666287, 75.8293395278, 3.27289462581])
            p = fix
            # cov_p = nclt_gnss_var
            cov_p = np.ones(3) * 0.001

            v = np.zeros(3)
            cov_v = np.ones(3) * 0.001

            ab = np.array([0.0, 0.0, 0.0])
            cov_ab = aw_var * np.ones(3)

            wb = np.zeros(3)
            cov_wb = ww_var * np.ones(3)

            t = time

            kf.initialize_state(p=p, cov_p=cov_p, v=v, cov_v=cov_v, ab=ab, cov_ab=cov_ab, wb=wb, cov_wb=cov_wb, time=t)

    elif gnss.fix_mode == 2:
        if kf.is_initialized():
            def hx_func(state):
                Hx = np.zeros([2, 16])
                Hx[:, 0:2] = np.eye(2)
                return Hx

            def meas_func(state):
                return fix[0:2] - state[0:2]

            V = np.diag(nclt_gnss_var[0:2])
            kf.correct(meas_func, hx_func, V, time, measurementname='gnss_no_alt')

            publish_state(state_pub)
            publish_gnss(converted_gnss_pub, time, fix)


def imu_callback(data):
    am = NCLTDataConversions.vector_ned_to_enu(np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))

    global latest_acceleration
    latest_acceleration = am

    am = np.array([am[0],am[1],am[2]])

    wm = NCLTDataConversions.vector_ned_to_enu(np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z]))

    time = data.header.stamp.to_sec()

    if kf.is_initialized():
        kf.predict(am, am_var, wm, wm_var, time, inputname='imu')
        publish_state(state_pub)


def mag_callback(data):
    mm = NCLTDataConversions.vector_ned_to_enu(np.array([data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z]))
    time = data.header.stamp.to_sec()

    ori = get_orientation_from_magnetic_field(mm, latest_acceleration)

    if kf.is_initialized():

        # def hx_func(state):
        #     Hx = np.zeros([4,16])
        #     Hx[:,6:10] = np.eye(4)
        #     return Hx

        def meas_func(state):
            q_ori = np.concatenate([ori[1:4], [ori[0]]])
            q_stt = np.concatenate([state[7:10], [state[6]]])
            # q_err = tft.quaternion_multiply(q_ori, tft.quaternion_conjugate(q_stt))

            e_ori = np.array(tft.euler_from_quaternion(q_ori))
            e_stt = np.array(tft.euler_from_quaternion(q_stt))
            e_err = e_ori - e_stt
            return e_err

        def hx_func(state):
            def q2e(astate):
                e = tft.euler_from_quaternion((astate[7], astate[8], astate[9], astate[6]))
                return np.array(e)
            j = nd.Jacobian(q2e)
            return j(state)


        V = np.diag(np.ones(3) * 1e-1)
        kf.correct(meas_func, hx_func, V, time, measurementname='magnetometer')
        publish_state(state_pub)
        publish_magnetic(converted_mag_pub, ori, time)

    else:
        v = np.zeros(3)
        cov_v = np.ones(3) * 0.001

        q = ori
        cov_q = np.ones(3) * 0.001

        ab = np.zeros(3)
        cov_ab = aw_var * np.ones(3)

        wb = np.zeros(3)
        cov_wb = ww_var * np.ones(3)

        kf.initialize_state(v=v, cov_v=cov_v, q=q, cov_q=cov_q, ab=ab, cov_ab=cov_ab, wb=wb, cov_wb=cov_wb, g=g,
                            time=time)

        if kf.is_initialized():
            log('State initialized.')


def laser_dt_callback(data):
    time = data.header.stamp.to_sec()
    dx = data.pose.pose.position.x
    dy = data.pose.pose.position.y
    dz = data.pose.pose.position.z
    dp_laser = np.array([dx, dy, dz])

    global laser_dt_prev_time
    if laser_dt_prev_time == -1:
        laser_dt_prev_time = time
    else:
        def h(state):
            return state[16:19] - state[0:3]

        def meas_func(state):
            dp_state = h(state)
            error = np.array(dp_laser - dp_state)

            print 'dp_laser:', dp_laser
            print 'dp_state:', dp_state
            print 'error:', error

            return error

        def hx_func(state):
            jh = nd.Jacobian(h)
            return jh(state)

        V = np.diag(np.ones(3) * 0.1)

        print 'ld_prev_time:', laser_dt_prev_time, 'ld_new_time:', time
        kf.correct_relative(meas_func, hx_func, V, laser_dt_prev_time, time, measurementname='laser_dt')
        laser_dt_prev_time = time


def laser_callback(data):
    points = np.array(data.data)
    num_values = points.shape[0]
    num_fields = 5
    num_points = num_values / num_fields

    pc2_msg = PointCloud2()
    pc2_msg.header.stamp = rospy.Time.now()
    pc2_msg.header.frame_id = 'base_link'
    pc2_msg.height = 1
    float_size = 4
    pc2_msg.width = num_values * float_size
    pc2_msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('l', 16, PointField.FLOAT32, 1),
    ]
    pc2_msg.is_bigendian = False
    pc2_msg.point_step = num_fields * float_size
    pc2_msg.row_step = pc2_msg.point_step * num_points
    pc2_msg.is_dense = False
    pc2_msg.width = num_points
    pc2_msg.data = np.asarray(points, np.float32).tostring()
    pointcloud_pub.publish(pc2_msg)


def gnss_mod_callback(data):
    time = data.header.stamp.to_sec()
    if time > 1357847254.9 and time < 1357847287.7:
        pass
    else:
        gnss_mod_pub.publish(data)


if __name__ == '__main__':
    rospy.init_node(Constants.LOCATOR_NODE_NAME, anonymous=True)
    log('Node initialized.')
    rospy.Subscriber('raw_data/gnss_fix', NavSatFix, gnss_mod_callback, queue_size=1)
    rospy.Subscriber('raw_data/gnss_fix_mod', NavSatFix, gnss_callback, queue_size=1)


    rospy.Subscriber('raw_data/imu', Imu, imu_callback, queue_size=1)
    rospy.Subscriber('raw_data/magnetometer', MagneticField, mag_callback, queue_size=1)
    rospy.Subscriber('/velodyne_packet', Float64MultiArray, laser_callback, queue_size=1)
    rospy.Subscriber('/laser_odom_dt', Odometry, laser_dt_callback, queue_size=1)

    state_pub = rospy.Publisher(Constants.STATE_TOPIC, State, queue_size=1)
    converted_gnss_pub = rospy.Publisher('/converted_data/gnss', State, queue_size=1)
    converted_mag_pub = rospy.Publisher('/converted_data/magnetic', State, queue_size=1)
    pointcloud_pub = rospy.Publisher('/raw_data/velodyne_points', PointCloud2, queue_size=1)

    gnss_mod_pub = rospy.Publisher('raw_data/gnss_fix_mod', NavSatFix, queue_size=1)

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

    tf2_static_broadcaster.sendTransform([world2map_static_tf, map2odom_static_tf])

    log('Locator ready.')
    rospy.spin()
