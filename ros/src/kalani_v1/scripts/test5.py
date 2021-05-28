import numpy as np
from utilities import *
from matplotlib import pyplot as plt
from kitti_datahandle import *
from sensor_msgs.msg import Imu, MagneticField, PointCloud2, Image
import rospy
import pcl

kd = KITTIData()
kd.load_data(calibrations=True)

print kd.calibrations.VEHICLE_R_STEREO