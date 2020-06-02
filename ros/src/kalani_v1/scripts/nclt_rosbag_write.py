#!/usr/bin/env python

import rosbag, rospy
import pandas as pd
import numpy as np
from kalani_v1.msg import GNSS, IMU, State
from constants import Constants
from filter.rotations_v1 import Quaternion

gps = pd.read_csv(Constants.GNSS_DATA_PATH, header=None)
gps.columns = ['time', 'modes', 'num_satss', 'lats', 'lngs', 'alts', 'tracks', 'speeds']

ms25 = pd.read_csv(Constants.IMU_DATA_PATH, header=None)
ms25.columns = ['time', 'mag_xs', 'mag_ys', 'mag_zs', 'accel_xs', 'accel_ys', 'accel_zs', 'rot_rs', 'rot_ps', 'rot_hs']

groundtruth = pd.read_csv(Constants.GROUNDTRUTH_DATA_PATH, header=None)
groundtruth.columns = ['time', 'pos_x', 'pos_y', 'pos_z', 'ori_r', 'ori_p', 'ori_y']

df_s = gps.append(ms25, ignore_index=True)
df_s = df_s.append(groundtruth, ignore_index=True)

df_s.sort_values(by=['time'], inplace=True)
df_s = (df_s.reset_index(drop=True)).fillna(True)
bag = rosbag.Bag(Constants.NCLT_SENSOR_DATA_ROSBAG_PATH, 'w')
i_max = len(df_s)


def rosbag():
    i = 0
    while i < i_max:
        if df_s['lats'][i] != True and df_s['lngs'][i] != True:
            timestamp = rospy.Time.from_sec(df_s['time'][i] / 1e6)
            gnss = GNSS()
            gnss.header.stamp = timestamp
            gnss.header.frame_id = Constants.GNSS_FRAME
            gnss.latitude = df_s['lats'][i]
            gnss.longitude = df_s['lngs'][i]
            gnss.altitude = df_s['alts'][i]
            gnss.fix_mode = df_s['modes'][i]
            bag.write(Constants.GNSS_DATA_TOPIC, gnss, t=timestamp)

        elif df_s['mag_xs'][i] != True and df_s['accel_xs'][i] != True:
            timestamp = rospy.Time.from_sec(df_s['time'][i] / 1e6)
            ms25 = IMU()
            ms25.header.stamp = timestamp
            ms25.header.frame_id = Constants.IMU_FRAME
            ms25.magnetic_field.x, ms25.magnetic_field.y, ms25.magnetic_field.z = [df_s['mag_xs'][i], df_s['mag_ys'][i],
                                                                                   df_s['mag_zs'][i]]
            ms25.linear_acceleration.x, ms25.linear_acceleration.y, ms25.linear_acceleration.z = [df_s['accel_xs'][i],
                                                                                                  df_s['accel_ys'][i],
                                                                                                  df_s['accel_zs'][i]]
            ms25.angular_velocity.x, ms25.angular_velocity.y, ms25.angular_velocity.z = [df_s['rot_rs'][i],
                                                                                         df_s['rot_ps'][i],
                                                                                         df_s['rot_hs'][i]]
            bag.write(Constants.IMU_DATA_TOPIC, ms25, t=timestamp)

        elif df_s['pos_x'][i] != True and df_s['ori_r'][i] != True:
            timestamp = rospy.Time.from_sec(df_s['time'][i] / 1e6)
            gt = State()
            gt.header.stamp = timestamp
            gt.header.frame_id = Constants.STATE_FRAME
            gt.position.x, gt.position.y, gt.position.z = [df_s['pos_x'][i], df_s['pos_y'][i], df_s['pos_z'][i]]
            q = Quaternion(euler=np.array([df_s['ori_r'][i],df_s['ori_p'][i],df_s['ori_y'][i]])).to_numpy().tolist()
            gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z = q
            bag.write(Constants.GROUNDTRUTH_DATA_TOPIC, gt, t=timestamp)

        else:
            print('something wrong in column %s', i)

        i += 1


if __name__ == '__main__':
    try:
        rosbag()
    except rospy.ROSInterruptException:
        pass
    finally:
        bag.close()