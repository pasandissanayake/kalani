#!/usr/bin/env python

import rosbag
import rospy
import time
from kalani_v1.msg import GNSS,IMU
from constants import Constants


bag=rosbag.Bag(Constants.NCLT_SENSOR_DATA_ROSBAG)
rospy.init_node(Constants.NCLT_SENSOR_DATA_ROSBAG_NODE_NAME, anonymous=True)
pub_gnss = rospy.Publisher(Constants.GNSS_DATA_TOPIC, GNSS, queue_size=10)
pub_ms25 = rospy.Publisher(Constants.IMU_DATA_TOPIC, IMU, queue_size=10)


def log(message):
    rospy.loginfo(Constants.NCLT_SENSOR_DATA_ROSBAG_NODE_NAME + ' := ' + str(message))


def ros_read():
    dt=0
    arg=False
    for topic, msg, t in bag.read_messages(topics=[Constants.IMU_DATA_TOPIC,Constants.GNSS_DATA_TOPIC]):
        if arg:
            dt=(t.to_sec()-t_prev.to_sec())

        t_prev=t
        time.sleep(dt)

        if topic==Constants.GNSS_DATA_TOPIC:
            pub_gnss.publish(msg)
        else:
            pub_ms25.publish(msg)
            arg=True

if __name__ == '__main__':
    try:
        log('Started sending data.')
        ros_read()
        log('Finished sending data.')
    except rospy.ROSInterruptException:
        log('Node stopped ungracefully.')
    finally:
       bag.close()




