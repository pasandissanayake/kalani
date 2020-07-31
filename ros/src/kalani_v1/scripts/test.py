# import numpy as np
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# from datasetutils.nclt_data_conversions import *
import sys
sys.path.append('../')
from constants import Constants
from filter.rotations_v1 import Quaternion
#
#
# dataset = NCLTData(Constants.NCLT_DATASET_DIRECTORY)
#
# gt_interpol_x = interp1d(dataset.groundtruth.time,dataset.groundtruth.x,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
# gt_interpol_y = interp1d(dataset.groundtruth.time,dataset.groundtruth.y,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
# gt_interpol_h = interp1d(dataset.groundtruth.time,dataset.groundtruth.h,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
#
# imu = np.loadtxt(Constants.NCLT_DATASET_DIRECTORY + '/' + Constants.NCLT_GNSS_DATA_FILE_NAME, delimiter=',')
# ori = gt_interpol_h(imu[:,0])
# vel = np.array([[imu[i,0],(imu[i,4]-imu[i-1,4])/(imu[i,0]-imu[i-1,0]),(imu[i,5]-imu[i-1,5])/(imu[i,0]-imu[i-1,0])] for i in range(1,len(imu))])
# acc = np.array([[vel[i,0],(vel[i,1]-vel[i-1,1])/(vel[i,0]-vel[i-1,0]),(vel[i,2]-vel[i-1,2])/(vel[i,0]-vel[i-1,0])] for i in range(1,len(vel))])
#
# pos = np.zeros((len(acc),3))
# pos = np.array([acc[i,0], pos[i-1,1] + vel[i+]])
# plt.plot(dataset.groundtruth.x,dataset.groundtruth.y)
# plt.show()


from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

imu = np.loadtxt(Constants.NCLT_DATASET_DIRECTORY + '/' + 'ms25.csv', delimiter=',')
gt = np.loadtxt(Constants.NCLT_DATASET_DIRECTORY + '/' + Constants.NCLT_GROUNDTRUTH_DATA_FILE_NAME, delimiter=',')
gt_inerpol = interp1d(gt[:,0],gt[:,1:7],axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')


AccX_Variance = 0.0000020
Time = imu[:,0]
RefPosX = gt_inerpol(Time)[:,0]

yaw = gt_inerpol(Time)[:,5]
AccX_Value = imu[:,4]*np.cos(yaw) - imu[:,5] * np.sin(yaw)
print np.average(AccX_Value)

Time = (Time - Time[0]) / 1e6

# time step
dt = (imu[1,0] - imu[0,0]) / 1e6

# transition_matrix
F = [[1, dt, 0.5*dt**2],
     [0,  1,       dt],
     [0,  0,        1]]

# observation_matrix
H = [0, 0, 1]

# transition_covariance
Q = [[  0,    0,      0],
     [  0,  0.0,      0],
     [  0,    0,      10e-6]]

# observation_covariance
R = AccX_Variance

# initial_state_mean
X0 = [RefPosX[0],
      0,
      AccX_Value[0]]

# initial_state_covariance
P0 = [[  0,    0,               0],
      [  0,    0,               0],
      [  0,    0,   AccX_Variance]]

n_timesteps = AccX_Value.shape[0]
n_dim_state = 3
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

kf = KalmanFilter(transition_matrices = F,
                  observation_matrices = H,
                  transition_covariance = Q,
                  observation_covariance = R,
                  initial_state_mean = X0,
                  initial_state_covariance = P0)

# iterative estimation for each new measurement
for t in range(n_timesteps):
    if t == 0:
        filtered_state_means[t] = X0
        filtered_state_covariances[t] = P0
    else:
        filtered_state_means[t], filtered_state_covariances[t] = (
        kf.filter_update(
            filtered_state_means[t-1],
            filtered_state_covariances[t-1],
            AccX_Value[t]
        )
    )



f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(Time, AccX_Value, "b-", label="Input AccX")
axarr[0].plot(Time, filtered_state_means[:, 2], "r-", label="Estimated AccX")
axarr[0].set_title('Acceleration X')
axarr[0].grid()
axarr[0].legend()
axarr[0].set_ylim([-4, 4])

# axarr[1].plot(Time, RefVelX, label="Reference VelX")
# axarr[1].plot(Time, filtered_state_means[:, 1], "r-", label="Estimated VelX")
# axarr[1].set_title('Velocity X')
# axarr[1].grid()
# axarr[1].legend()
# axarr[1].set_ylim([-1, 20])

axarr[1].plot(Time, RefPosX, "b-", label="Reference PosX")
axarr[1].plot(Time, filtered_state_means[:, 0], "r-", label="Estimated PosX")
axarr[1].set_title('Position X')
axarr[1].grid()
axarr[1].legend()

plt.show()