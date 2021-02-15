import numpy as np
from utilities import *
from matplotlib import pyplot as plt
from kaist_datahandle import *
from pyproj import Proj
from scipy.interpolate import interp1d
import time
import tf.transformations as tft
from mpl_toolkits import mplot3d

kd = KAISTData()
kd.load_data(sequence='urban28', groundtruth=True, calibrations=True)
vo = np.loadtxt('/home/entc/Desktop/a.txt', delimiter=' ')
t = vo[:,0]
gt = np.array([kd.groundtruth.interp_x(t), kd.groundtruth.interp_y(t), kd.groundtruth.interp_z(t)]).T
gt[:,0] = gt[:,0] - gt[0,0]
gt[:,1] = gt[:,1] - gt[0,1]
gt[:,2] = gt[:,2] - gt[0,2]
w_R_b = tft.quaternion_matrix(tft.quaternion_from_euler(kd.groundtruth.interp_r(t[0]), kd.groundtruth.interp_p(t[0]), kd.groundtruth.interp_h(t[0])))[0:3,0:3]
b_R_c = kd.calibrations.VEHICLE_R_STEREO[0:3,0:3]
c_R_w = np.matmul(b_R_c.T,w_R_b.T)
print gt.shape
for i in range(len(gt)):
    gt[i] = np.matmul(c_R_w, gt[i])


fig = plt.figure()
ax = plt.axes(projection='3d')


#ax.plot3D(gt[:,0],gt[:,1], label='vo2')
ax.plot3D(vo[:,1],vo[:,2],vo[:,3], label='vo3')
plt.legend()
plt.show()
