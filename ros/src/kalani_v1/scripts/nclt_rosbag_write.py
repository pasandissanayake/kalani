#!/usr/bin/env python

import rosbag, rospy
import pandas as pd
from kalani_v1.msg import GNSS,IMU
from constants import Constants


gps = pd.read_csv(Constants.GNSS_DATA_PATH,header=None)
ms25 = pd.read_csv(Constants.IMU_DATA_PATH,header=None)

utimes_g = list(gps[0])
modes = list(gps[1])
num_satss = list(gps[2])
lats = list(gps[3])
lngs = list(gps[4])
alts = list(gps[5])
tracks = list(gps[6])
speeds = list(gps[7])

utimes_m= list(ms25[0])

mag_xs = list(ms25[1])
mag_ys = list(ms25[2])
mag_zs = list(ms25[3])

accel_xs = list(ms25[4])
accel_ys = list(ms25[5])
accel_zs = list(ms25[6])

rot_rs = list(ms25[7])
rot_ps = list(ms25[8])
rot_hs = list(ms25[9])
bag = rosbag.Bag(Constants.NCLT_SENSOR_DATA_ROSBAG, 'w')
i_max=len(gps)
j_max=len(ms25)


def log(message):
    rospy.loginfo(Constants.NCLT_SENSOR_DATA_ROSBAG_NODE_NAME + ' := ' + str(message))


def rosbag():
    i=0
    j=0
    while i<i_max or j<j_max :
        if i<i_max:
            time_g=utimes_g[i]
        else:
            time_g=utimes_g[i-1]
        if j<j_max:
            time_m=utimes_m[j]
        else:
            time_m=utimes_m[j-1]

        if (time_g>=time_m or j==j_max) and (i<i_max):
            timestamp = rospy.Time.from_sec(utimes_g[i]/1e6)
            gnss = GNSS()
            gnss.header.stamp = timestamp
            gnss.header.frame_id = Constants.GNSS_FRAME
            gnss.latitude = lats[i]
            gnss.longitude = lngs[i]
            gnss.altitude = alts[i]
            gnss.fix_mode = modes[i]
            bag.write(Constants.GNSS_DATA_TOPIC,gnss, t=timestamp)
            i+=1

        if (time_g<=time_m or i==i_max) and (j<j_max):
            timestamp = rospy.Time.from_sec(utimes_m[j]/1e6)
            ms25=IMU()
            ms25.header.stamp =timestamp
            ms25.header.frame_id = Constants.IMU_FRAME
            ms25.magnetic_field.x, ms25.magnetic_field.y, ms25.magnetic_field.z = [ x * 10 ** (-4) for x in [mag_xs[j],mag_ys[j],mag_zs[j]] ]
            ms25.linear_acceleration.x, ms25.linear_acceleration.y, ms25.linear_acceleration.z = [accel_xs[j],accel_ys[j],accel_zs[j]]
            ms25.angular_velocity.x, ms25.angular_velocity.y, ms25.angular_velocity.z = [rot_rs[j],rot_ps[j],rot_hs[j]]
            bag.write(Constants.IMU_DATA_TOPIC,ms25, t=timestamp)
            j+=1


if __name__ == '__main__':
    try:
        log('Started writing rosbag.')
        rosbag()
        log('Finished writing rosbag.')
    except rospy.ROSInterruptException:
        log('Script stopped ungracefully.')
    finally:
        bag.close()
