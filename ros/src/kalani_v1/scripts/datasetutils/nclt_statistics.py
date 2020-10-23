import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from nclt_data_conversions import *
sys.path.append('../')
from constants import Constants

datadir = '/home/pasan/kalani-data/nclt/2013-01-10'
dataset = NCLTData(datadir)

# gnss plots

time = dataset.converted_gnss.time
gt_interpol_x = interp1d(dataset.groundtruth.time,dataset.groundtruth.x,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
error_x = dataset.converted_gnss.x - gt_interpol_x(time)
mean_x = np.average(error_x)
variance_x = np.average(error_x**2)

gt_interpol_y = interp1d(dataset.groundtruth.time,dataset.groundtruth.y,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
error_y = dataset.converted_gnss.y - gt_interpol_y(time)
mean_y = np.average(error_y)
variance_y = np.average(error_y**2)

gt_interpol_z = interp1d(dataset.groundtruth.time,dataset.groundtruth.z,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
error_z = dataset.converted_gnss.z[dataset.converted_gnss.fix_mode==3] - gt_interpol_z(time[dataset.converted_gnss.fix_mode==3])
mean_z = np.average(error_z)
variance_z = np.average(error_z**2)

# non-zero mean gnss error distributions
fig1, (ax11, ax12, ax13) = plt.subplots(1, 3)
fig1.suptitle('Non-zero mean GNSS error distributions')
num_bins = 100
error_range = np.arange(-15,15,0.1)

ax11.hist(error_x, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax11.plot(error_range, mlab.normpdf(error_range, mean_x, np.sqrt(variance_x)))
ax11.set_xlabel('x (m)\n var x:' + str(variance_x) + '\nmean x:' + str(mean_x))

ax12.hist(error_y, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax12.plot(error_range, mlab.normpdf(error_range, mean_y, np.sqrt(variance_y)))
ax12.set_xlabel('y (m)\n var y:' + str(variance_y) + '\nmean y:' + str(mean_y))

z_error_range = np.arange(-200,200,1)
ax13.hist(error_z, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax13.plot(z_error_range, mlab.normpdf(z_error_range, mean_z, np.sqrt(variance_z)))
ax13.set_xlabel('z (m)\n var z:' + str(variance_z) + '\nmean z:' + str(mean_z))

# non-zero mean gnss 1d plots
fig2, ax21 = plt.subplots(1, 1)
fig2.suptitle('Non-zero mean GNSS 1D plots')

ax21.plot(dataset.converted_gnss.x, dataset.converted_gnss.y, marker='.', linestyle='None')
ax21.plot(dataset.groundtruth.x, dataset.groundtruth.y)



# # accelerations
# imu_raw = np.loadtxt('{}/ms25.csv'.format(datadir), delimiter=',')
# imu = np.zeros((len(imu_raw), 4))
# imu[:,0] = imu_raw[:,0] * 1e-6
# imu[:,1] = imu_raw[:,5]
# imu[:,2] = imu_raw[:,4]
# imu[:,3] = -imu_raw[:,6]
#
# end_idx = (np.abs(imu[:,0] - 1335704128.61069)).argmin()
# print imu[end_idx, 0]
#
# def k_th_difference(array, k):
#     reslen = array.shape[0] - k
#     result = np.zeros((reslen, 4))
#     result[:, 0] = array[:reslen, 0]
#     result[:, 1] = np.diff(array[:, 1], n=k)
#     result[:, 2] = np.diff(array[:, 2], n=k)
#     result[:, 3] = np.diff(array[:, 3], n=k)
#     return result
#
# acc_x_mean = np.mean(imu[:end_idx, 1])
# acc_y_mean = np.mean(imu[:end_idx, 2])
# acc_z_mean = np.mean(imu[:end_idx, 3])
#
# fig3, (ax31, ax32, ax33) = plt.subplots(3,1)
# fig3.suptitle('acceleration analysis')
#
# ax31.plot(imu[:end_idx, 0], imu[:end_idx, 1] - acc_x_mean, marker='*', label='x')
# ax31.plot(imu[:end_idx, 0], imu[:end_idx, 2] - acc_y_mean, marker='*', label='y')
# ax31.plot(imu[:end_idx, 0], imu[:end_idx, 3] - acc_z_mean, marker='*', label='z')
# ax31.text(0.05, 0.95, 'means:\nx = {}\ny = {}\nz = {}'.format(acc_x_mean, acc_y_mean, acc_z_mean),
#         verticalalignment='top', horizontalalignment='left',
#         transform=ax31.transAxes,
#         color='green', fontsize=8)
# ax31.legend()
#
# dif1 = k_th_difference(imu[:end_idx], 1)
# ax32.plot(dif1[:, 0], dif1[:, 1], label='x')
# ax32.plot(dif1[:, 0], dif1[:, 2], label='y')
# ax32.plot(dif1[:, 0], dif1[:, 3], label='z')
# ax32.set_title('first differences')
# ax32.legend()
#
# var_am_x = 0.5 * np.var(dif1[:, 1])
# var_aw_x = []
# for k in range(1,80,1):
#     difk = k_th_difference(imu[:end_idx], k)
#     var_aw_x.append((np.var(difk[:, 1]) - 2 * var_am_x) / k)
#
# ax33.plot(range(len(var_aw_x)), var_aw_x)
# ax33.text(0.05, 0.95, 'var am x = {}\nvar aw x = {}'.format(var_am_x, var_aw_x),
#         verticalalignment='top', horizontalalignment='left',
#         transform=ax33.transAxes,
#         color='green', fontsize=8)

plt.show()

