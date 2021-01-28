import numpy as np
from utilities import *
from matplotlib import pyplot as plt
from kaist_datahandle import *
from pyproj import Proj
from scipy.interpolate import interp1d
import time

start = time.time()
kd = KAISTData()
print 'loading started'
sequence = 'urban17'
kd.load_data(sequence=sequence, groundtruth=True, gnss=True, imu=False, altitude=False, vlpleft=False)
print 'loading done. time elapsed: {} s'.format(time.time() - start)
print 'start time: {:.9f} s'.format(kd.groundtruth.time[0],)

gps_dif_x = np.diff(kd.gnss.x)
gps_dif_y = np.diff(kd.gnss.y)
gps_dif = np.sqrt(gps_dif_x**2 + gps_dif_y**2)

gt_dif_x = np.diff(kd.groundtruth.interp_x(kd.gnss.time))
gt_dif_y = np.diff(kd.groundtruth.interp_y(kd.gnss.time))
gt_dif = np.sqrt(gt_dif_x**2 + gt_dif_y**2)

dif_dif = gps_dif - gt_dif

error_x = kd.gnss.x - kd.groundtruth.interp_x(kd.gnss.time)
error_y = kd.gnss.y - kd.groundtruth.interp_y(kd.gnss.time)
error = np.sqrt(error_x**2 + error_y**2)

error_dif = np.diff(error)

fig1, (ax1, ax2) = plt.subplots(2,1)

fig1.suptitle('Sequence: {}'.format(sequence))

ax1.set_title('(distance between current, previous GPS) - (distance between current, previous GT) (m)')
ax1.plot(kd.gnss.time[:-1], dif_dif, label='Difference')
ax1.plot(kd.gnss.time[:-1], np.zeros(len(kd.gnss.time)-1))
ax1.plot(kd.gnss.time[:-1], np.ones(len(kd.gnss.time)-1) * 1, 'r--')
ax1.plot(kd.gnss.time[:-1], np.ones(len(kd.gnss.time)-1) * -1, 'r--')

ax2.set_title('(distance between current GPS and GT) - (distance between previous GPS and GT) (m)')
ax2.plot(kd.gnss.time[:-1], error_dif, label='Difference')
ax2.plot(kd.gnss.time[:-1], np.zeros(len(kd.gnss.time)-1))
ax2.plot(kd.gnss.time[:-1], np.ones(len(kd.gnss.time)-1) * 2, 'r--')
ax2.plot(kd.gnss.time[:-1], np.ones(len(kd.gnss.time)-1) * -2, 'r--')

ax1.legend()
ax2.legend()
ax1.grid()
ax2.grid()
ax1.margins(0.05)
ax2.margins(0.05)
plt.show()