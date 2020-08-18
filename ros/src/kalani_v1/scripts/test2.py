import numpy as np
import matplotlib.pyplot as plt

imu = np.loadtxt('/home/entc/kalani/data/nclt/2012-04-29/ms25.csv', delimiter=',')
gt = np.loadtxt('/home/entc/kalani/data/nclt/2012-04-29/groundtruth.csv', delimiter=',')

plt.plot(imu[0:2000,0], imu[0:2000,4], label='imu-x')

gt_x_norm = gt[:,0:2]
gt_x_norm[:,1] = gt_x_norm[:,1] * np.max(np.abs(imu[:,4])) / np.max(np.abs(gt_x_norm[:,1]))
plt.plot(gt_x_norm[0:6000,0], gt_x_norm[0:6000,1], label='gt-x')

plt.legend()
plt.grid()
plt.show()
