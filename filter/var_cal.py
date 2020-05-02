import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from .rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

# Data directory
data_dir = "2012-04-29"



# Rotation matrix from NED to ENU
R_ned_enu = np.array([[0,1,0],[1,0,0],[0,0,-1]])





# Load ground truth data in ENU (East-North-Up) frame [:,0]=time, 1=East, 2=North, 3=Up, 4=roll(around East), 5=pitch(around North), 6=yaw(around Up)
gt_raw = np.loadtxt(data_dir + "/" + "groundtruth.csv", delimiter=",") #0=time, 1=North, 2=East, 3=Down, 4=roll(around North), 5=pitch(around East), 6=yaw(around Down)
gt = np.zeros(gt_raw.shape)
gt[:,0] = gt_raw[:,0]
gt[:,1:4] = (R_ned_enu @ gt_raw[:,1:4].T).T
gt[:,4:7] = (R_ned_enu @ gt_raw[:,4:7].T).T







# Load GNSS data
gnss_raw = np.loadtxt(data_dir + "/" + "gps.csv", delimiter=",")
gnss = []
for i in range(gnss_raw.shape[0]):
    rec = gnss_raw[i]
    if str(rec[5]) != 'nan':
        gnss.append([rec[0],rec[3],rec[4],rec[5],rec[1]])
    else:
        # gnss.append([rec[0], rec[3], rec[4], gnss_raw[i-1,5], rec[1]])
        gnss.append([rec[0], rec[3], rec[4], 270, rec[1]])
gnss = np.array(gnss)

# GNSS coordinates of the origin
lat0 = 42.293227 * np.pi / 180
lng0 = -83.709657 * np.pi / 180
alt0 = 270

lat = gnss[:,1]
lng = gnss[:,2]
alt = gnss[:,3]

dLat = lat - lat0
dLng = lng - lng0
dAlt = alt0 - alt

# GNSS data in NED
r = 6400000
gnss[:,1] = r * np.sin(dLat)
gnss[:,2] = r * np.cos(lat0) * np.sin(dLng)
gnss[:,3] = dAlt

# GNSS data in ENU
gnss[:,1:4] = (R_ned_enu @ gnss[:,1:4].T).T

gt_function = interp1d(gt[:,0],gt[:,1:4], axis=0, bounds_error=False, kind='linear', fill_value='extrapolate')
gt_interpol = gt_function(gnss[:,0])
dif = (gnss[:,1:4]-gt_interpol[:,0:3])**2
print(dif)
var = np.sum(dif, axis=0) / gnss.shape[0]
print(var)
