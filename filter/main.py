import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from filter_v1 import Filter_V1
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

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

# # Load odometry data
# odom_raw = np.loadtxt(data_dir + "/" + "odometry_mu_100hz.csv", delimiter = ",")
# odom = np.zeros([odom_raw.shape[0],4])
# odom[:,0:4] = odom_raw[:,0:4]
# odom[:,1:4] = (R_ned_enu @ odom[:,1:4].T).T + np.tile(gnss[0,1:4],(odom_raw.shape[0],1))

# Load odometry data as velocity
odom_raw = np.loadtxt(data_dir + "/" + "odometry_mu_100hz.csv", delimiter = ",")
odom = np.zeros([odom_raw.shape[0]-1,4])
for i in range(1,odom_raw.shape[0]):
    odom[i-1,0] = odom_raw[i,0]
    odom[i-1,1:4] = (odom_raw[i,1:4]-odom_raw[i-1,1:4]) * 10**6 / (odom_raw[i,0]-odom_raw[i-1,0])
odom[:,1:4] = (R_ned_enu @ odom[:,1:4].T).T


# Load IMU data in body frame [:,0]=time, 1=mag_x,
# mag_x = ms25[:, 1]
# mag_y = ms25[:, 2]
# mag_z = ms25[:, 3]
# accel_x = ms25[:, 4]
# accel_y = ms25[:, 5]
# accel_z = ms25[:, 6]
# rot_r = ms25[:, 7]
# rot_p = ms25[:, 8]
# rot_y = ms25[:, 9]
imu = np.loadtxt(data_dir + "/" + "ms25.csv", delimiter=",")




#
# with open('data/pt3_data.pkl', 'rb') as file:
#     data = pickle.load(file)
#
# gt_raw = data['gt']
# imu_f_raw = data['imu_f']
# imu_w_raw = data['imu_w']
# gnss_raw = data['gnss']
# lidar_raw = data['lidar']
#
# gt = np.zeros([gt_raw.p.shape[0],7])
# gt[:,1:4] = gt_raw.p
# gt[:,4:7] = gt_raw.r
# gt[:,0] = gt_raw._t[0:gt_raw.p.shape[0]]
#
# gnss = np.zeros([gnss_raw.data.shape[0],4])
# gnss[:,0] = gnss_raw.t
# gnss[:,1:4] = gnss_raw.data
#
# imu = np.zeros([imu_f_raw.data.shape[0],10])
# imu[:,0] = imu_f_raw.t[0:imu_f_raw.data.shape[0]]
# imu[:,4:7] = imu_f_raw.data
# imu[:,7:10] = imu_w_raw.data

time_multiplication_factor = 10**(-6)



kf = Filter_V1()
p0 = gnss[0,1:4]
v0 = np.zeros(3)
q0 = Quaternion(euler=gt[0,4:7]).to_numpy()
g0 = [0, 0, -9.8]
ab0 = np.zeros(3)
wb0 = np.zeros(3)
x0 = np.array([*p0,*v0,*q0,*g0,*ab0,*wb0])
P0 = np.zeros([15,15])
t0 = imu[0,0]*time_multiplication_factor
kf.initialize_state(x0,P0,t0)
print([*p0,*v0,*q0,*ab0,*wb0])

min_time_imu = min(imu[:,0])
gnss_k = min(i for i in range(gnss.shape[0]) if gnss[i,0] > min_time_imu)
odom_k = min(i for i in range(odom.shape[0]) if odom[i,0] > min_time_imu)

pose_est = []
cov_est = []
bias_est = []

length = imu.shape[0]
length = 26000
for k in range(1,length):

    kf.predict(imu[k,4:7],imu[k,7:10],imu[k,0]*time_multiplication_factor)

    # GNSS correction

    # GNSS with altitude correction
    # if gnss_k < len(gnss[:, 0]) and gnss[gnss_k, 0] >= imu[k - 1, 0] and gnss[gnss_k, 0] <= imu[k, 0] and gnss[gnss_k,4] == 3:
    #     kf.correct(gnss[gnss_k,1:4],gnss[gnss_k,0]*time_multiplication_factor,Filter_V1.GNSS_WITH_ALT)
    #     gnss_k += 1

    # GNSS without altitude correction
    if gnss_k < len(gnss[:, 0]) and gnss[gnss_k, 0] >= imu[k - 1, 0] and gnss[gnss_k, 0] <= imu[k, 0] and gnss[gnss_k,4] >= 2:
        kf.correct(gnss[gnss_k,1:3],gnss[gnss_k,0]*time_multiplication_factor,Filter_V1.GNSS_NO_ALT)
        gnss_k += 1

    # Neglect other GNSS readings
    elif gnss_k < len(gnss[:, 0]) and (gnss[gnss_k, 0] >= imu[k - 1, 0] and gnss[gnss_k, 0] <= imu[k, 0] or gnss[gnss_k, 0]<imu[k-1, 0]):
        gnss_k += 1

    # Odometry correction

    # Odometry with altitude
    if odom_k < len(odom[:, 0]) and odom[odom_k, 0] >= imu[k - 1, 0] and odom[odom_k, 0] <= imu[k, 0]:
        kf.correct(odom[odom_k,1:4],odom[odom_k,0]*time_multiplication_factor,Filter_V1.ODOM_WITH_ALT)
        odom_k += 1

    # Neglect odometry readings if not within the time span of IMU readings
    while odom[odom_k, 0]<imu[k, 0]:
        odom_k += 1

    p,v,q,ab,wb,P = kf.get_state()

    pose_est.append([imu[k,0],*p,*q])
    cov_est.append([imu[k,0],P])
    bias_est.append([imu[k,0],*ab,*wb])

pose_est = np.array(pose_est)
cov_est = np.array(cov_est)
bias_est = np.array(bias_est)






# 3D plot
fig_estimate = plt.figure()
ax = fig_estimate.add_subplot(111, projection='3d')
ax.plot(pose_est[:,1], pose_est[:,2], pose_est[:,3], label='Estimate')
ax.plot(gt[:,1], gt[:,2], gt[:,3], label='Ground Truth')
# ax.plot(gnss[:,1], gnss[:,2], gnss[:,3], label='GNSS')
# ax.plot(odom[:,1], odom[:,2], odom[:,3], label='Odometry')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
# ax.set_xlim(-600, 200)
# ax.set_ylim(-100, 200)
ax.set_zlim(-2, 2)
ax.set_title('Estimate')
ax.legend()
ax.view_init(elev=45, azim=-50)

# Error plots
fig_error, ax = plt.subplots(2, 3)
fig_error.suptitle('Error Plots')
# Convert estimated quaternions to euler angles
p_est_euler = []
p_cov_euler_std = []
p_cov_std = []
for i in range(pose_est.shape[0]):
    qc = Quaternion(*pose_est[i,4:8])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ cov_est[i,1][6:9,6:9] @ J.T)))

    # Covariance for position
    p_cov_std.append(np.sqrt(np.diagonal(cov_est[i, 1][0:3, 0:3])))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)
p_cov_std = np.array(p_cov_std)

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
gt_function = interp1d(gt[:,0],gt[:,1:7], axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
gt_interpol = gt_function(pose_est[:,0])
for i in range(0,3):
    ax[0, i].plot(pose_est[:,0], gt_interpol[:,i] - pose_est[:, i+1])
    ax[0, i].plot(pose_est[:,0],  3 * p_cov_std[:, i], 'r--')
    ax[0, i].plot(pose_est[:,0], -3 * p_cov_std[:, i], 'r--')
    ax[0, i].set_title(titles[i])
    ax[0, i].set_ylim(-20,20)
    ax[0, i].axvline(x=gt[-1,0])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(pose_est[:,0], angle_normalize(gt_interpol[:,i+3] - p_est_euler[:, i]))
    ax[1, i].plot(pose_est[:,0],  3 * p_cov_euler_std[:, i], 'r--')
    ax[1, i].plot(pose_est[:,0], -3 * p_cov_euler_std[:, i], 'r--')
    ax[1, i].set_ylim(-10, 10)
    ax[1, i].set_title(titles[i+3])
    ax[1, i].axvline(x=gt[-1, 0])
ax[1,0].set_ylabel('Radians')

# 1D plots
fig_1d, ax = plt.subplots(1, 3)
fig_1d.suptitle('1D Plots')
for i in range(0,3):
    ax[i].plot(gt[:,0], gt[:,i+1], label='Ground truth')
    ax[i].plot(gnss[:, 0], gnss[:, i + 1], label='GNSS')
    # ax[i].plot(odom[:, 0], odom[:, i + 1], label='Odometry')
    ax[i].plot(pose_est[:,0], pose_est[:,i+1], label='Estimate')
    ax[i].set_title(titles[i])
ax[0].set_ylabel('Meters')
ax[0].legend()

# Bias plots
bias_titles = ['ab-x', 'ab-y', 'ab-z', 'wb-x', 'wb-y', 'wb-z']
fig_biases, ax = plt.subplots(2, 3)
fig_biases.suptitle('Biases')
for i in range(0,3):
    ax[0,i].plot(bias_est[:,0], bias_est[:,i+1])
    ax[1,i].plot(bias_est[:,0], bias_est[:,i+4])
    ax[0,i].set_title(bias_titles[i])
    ax[1,i].set_title(bias_titles[i+3])
    ax[0,i].set_ylabel('m/s^2')
    ax[1,i].set_ylabel('rad/s')
plt.show()
