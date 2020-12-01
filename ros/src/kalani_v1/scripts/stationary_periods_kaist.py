import numpy as np
import matplotlib.pyplot as plt
from datasetutils.nclt_data_conversions import *
from constants import *
from utilities import *
from kaist_datahandle import *


kd = KAISTData()
kd.load_data(gnss=False, altitude=False, vlpleft=False)






# # find stationary periods
#
# n_euc = np.sqrt(kd.groundtruth.interp_x(kd.imu.time)**2 + kd.groundtruth.interp_y(kd.imu.time)**2 + kd.groundtruth.interp_z(kd.imu.time)**2).T
#
# window_width = 20
# window_height = 0.01
#
# plt.plot(kd.imu.time, n_euc)
#
# i = 0
# while i < kd.imu.time.shape[0]:
#     start_index = i
#     start_time = kd.imu.time[start_index]
#     end_index = np.argmin(np.abs(kd.imu.time - start_time - window_width))
#     end_time = kd.imu.time[end_index]
#
#     if np.all(np.abs(n_euc[start_index:end_index] - n_euc[start_index]) < window_height):
#         plt.plot(kd.imu.time[start_index:end_index], n_euc[start_index:end_index], 'r+')
#         print start_index, end_index, start_time, end_time
#         i = end_index + 1
#     else:
#         i += 1





start_idx = 8421
end_idx = 12422+1

print 'first difference: {} s'.format(np.average(np.diff(kd.imu.time[start_idx:end_idx])))
print 'frequency: {} Hz'.format((end_idx - start_idx) / (kd.imu.time[end_idx]-kd.imu.time[start_idx]))

plt.plot(kd.imu.time[start_idx:end_idx], kd.imu.acceleration.x[start_idx:end_idx], 'b-')
plt.plot(kd.imu.time[start_idx:end_idx], kd.imu.acceleration.y[start_idx:end_idx], 'g-')
plt.plot(kd.imu.time[start_idx:end_idx], np.ones(len(kd.imu.acceleration.x[start_idx:end_idx]))*np.average(kd.imu.acceleration.x[start_idx:end_idx]), 'r-')
plt.plot(kd.imu.time[start_idx:end_idx], np.ones(len(kd.imu.acceleration.y[start_idx:end_idx]))*np.average(kd.imu.acceleration.y[start_idx:end_idx]), 'r-')

plt.grid()
plt.show()

np.savetxt("/home/pasan/Desktop/kaist_imu_forward.csv", kd.imu.acceleration.x[start_idx:end_idx]-np.average(kd.imu.acceleration.x[start_idx:end_idx]), delimiter=",")
np.savetxt("/home/pasan/Desktop/kaist_imu_rightward.csv", -kd.imu.acceleration.y[start_idx:end_idx]+np.average(kd.imu.acceleration.y[start_idx:end_idx]), delimiter=",")