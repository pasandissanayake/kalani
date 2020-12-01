import numpy as np
import matplotlib.pyplot as plt
from datasetutils.nclt_data_conversions import *
from constants import *

nd = NCLTData(Constants.NCLT_DATASET_DIRECTORY)






# # find stationary periods
#
# n_euc = np.sqrt(nd.groundtruth.x**2 + nd.groundtruth.y**2 + nd.groundtruth.z**2)
#
# window_width = 5
# window_height = 0.05
#
# plt.plot(nd.groundtruth.time, n_euc)
#
# i = 0
# while i < nd.groundtruth.length:
#     start_index = i
#     start_time = nd.groundtruth.time[start_index]
#     end_index = np.argmin(np.abs(nd.groundtruth.time - start_time - window_width))
#     end_time = nd.groundtruth.time[end_index]
#
#     if np.all(np.abs(n_euc[start_index:end_index] - n_euc[start_index]) < window_height):
#         plt.plot(nd.groundtruth.time[start_index:end_index], n_euc[start_index:end_index], 'r+')
#         print start_index, end_index, start_time, end_time
#         i = end_index + 1
#     else:
#         i += 1
#






imu_raw = np.loadtxt('/home/pasan/kalani-data/nclt/2013-01-10/ms25.csv', delimiter=',')
imu_raw[:, 0] = imu_raw[:, 0] * 1e-6
imu_stat = imu_raw[np.argmin(np.abs(imu_raw[:,0]-1357847340.254951)):np.argmin(np.abs(imu_raw[:,0]-1357847363.314949)), :]

print 'first difference: {} s'.format(np.average(np.diff(imu_stat[:,0])))
print 'frequency: {} Hz'.format(len(imu_stat) / (imu_stat[-1,0]-imu_stat[0,0]))

plt.plot(imu_stat[:,0], imu_stat[:,5], 'b--')
plt.plot(imu_stat[:,0], imu_stat[:,4], 'g--')
plt.plot(imu_stat[:,0], np.ones(len(imu_stat))*np.average(imu_stat[:,5]), 'r-')
plt.plot(imu_stat[:,0], np.ones(len(imu_stat))*np.average(imu_stat[:,4]), 'r-')

plt.grid()
plt.show()

np.savetxt("/home/pasan/Desktop/imu_forward_nonzero_mean.csv", imu_stat[:,4], delimiter=",")
np.savetxt("/home/pasan/Desktop/imu_rightward.csv", imu_stat[:,5]-np.average(imu_stat[:,5]), delimiter=",")
np.savetxt("/home/pasan/Desktop/imu_rightward_nonzero_mean.csv", imu_stat[:,5], delimiter=",")