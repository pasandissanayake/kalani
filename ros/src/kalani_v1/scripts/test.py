from datasetutils.kaist_data_conversions import KAISTDataConversions, KAISTData
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


kd = KAISTData('/home/entc/kalani/data/kaist/urban17')

gt_array = np.array([kd.groundtruth.time, kd.groundtruth.x, kd.groundtruth.y, kd.groundtruth.z]).T
gt_inter = interpolate.interp1d(gt_array[:, 0], gt_array[:, 1:4], axis=0, bounds_error=False, kind='linear', fill_value='extrapolate')
gt = gt_inter(kd.converted_gnss.time)
print gt
error = gt - np.array([kd.converted_gnss.x, kd.converted_gnss.y, kd.converted_gnss.z]).T

fig, ax = plt.subplots(1,3)
ax[0].hist(error[:,0], bins='auto')
ax[0].set_xlabel('x(m)')
ax[0].grid()
ax[1].hist(error[:,1], bins='auto')
ax[1].set_xlabel('y(m)')
ax[1].grid()
ax[1].set_title('GPS error distributions')
ax[2].hist(error[:,2], bins='auto')
ax[2].set_xlabel('z(m)')
ax[2].grid()
plt.show()

