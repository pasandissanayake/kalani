#!/usr/bin/env python

import rospy
import time
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import pandas as pd
import numpy as np
df=pd.read_csv('/home/chanaka/catkin_ws/src/beginner_tutorials/scripts/ms25_ros.csv')
def input_imu():
    pub=rospy.Publisher('imu_data', numpy_msg(Floats), queue_size=10)
    rospy.init_node('input_imu', anonymous=True)
    i_imu=0
    
    while not rospy.is_shutdown() and len(df)>i_imu:
	    
            imu_input=list(df.loc[i_imu])
            imu_input[0]=imu_input[0]*10**(-6)
            imu_input = np.array(imu_input, dtype=np.float32)
            pub.publish(imu_input)
            rospy.loginfo(imu_input)
	    print ('time_imu %s' % rospy.get_time())
            i_imu=i_imu+1
            if i_imu<len(df)-1:
               t=df.loc[i_imu+1][0]-df.loc[i_imu][0]
	    time.sleep(t*10**(-6))
            

if __name__ == '__main__':
    try:
       input_imu()
    except rospy.ROSInterruptException:
        pass
