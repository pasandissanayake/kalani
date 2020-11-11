import numpy as np
from utilities import *
from matplotlib import pyplot as plt
from kaist_datahandle import *
import time

start = time.time()
kd = KAISTData()
print 'loading started'
kd.load_data(groundtruth=True, gnss=True, imu=False, altitude=False, vlpleft=False)
print 'loading done. time elapsed: {} s'.format(time.time() - start)
print 'start time: {:.9f} s'.format(kd.groundtruth.time[0],)
# plt.plot(kd.groundtruth.time[:-1], np.diff(kd.groundtruth.x)/np.diff(kd.groundtruth.time), 'x', label='gt-x')
# plt.plot(kd.groundtruth.time[:-1], np.diff(kd.groundtruth.y)/np.diff(kd.groundtruth.time), 'x', label='gt-y')
# plt.plot(kd.groundtruth.time[:-1], np.diff(kd.groundtruth.z)/np.diff(kd.groundtruth.time), 'x', label='gt-z')
# r = range(len(kd.groundtruth.time)-1)
# plt.plot(r, np.diff(kd.groundtruth.x)/np.diff(kd.groundtruth.time), label='gt-x')
# plt.plot(r, np.diff(kd.groundtruth.y)/np.diff(kd.groundtruth.time), label='gt-y')
# plt.plot(r, np.diff(kd.groundtruth.z)/np.diff(kd.groundtruth.time), label='gt-z')

n = np.zeros(kd.gnss.x.shape[0])
std = np.sqrt(5)
n[0] = kd.groundtruth.interp_x(kd.gnss.time[0])
n[1] = kd.groundtruth.interp_x(kd.gnss.time[1])
n[2] = kd.groundtruth.interp_x(kd.gnss.time[2])
n[3] = kd.groundtruth.interp_x(kd.gnss.time[3])
for i in range(4,len(n)):
    n[i] = kd.gnss.x[i] + (kd.gnss.x[i-1] - n[i-1]) - 0.3 * (kd.gnss.x[i-2] - n[i-2] - kd.gnss.x[i-3] + n[i-3]) - 0.3 * (kd.gnss.x[i-3] - n[i-3] - kd.gnss.x[i-4] + n[i-4])
bins = 20
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(n-kd.groundtruth.interp_x(kd.gnss.time), bins=bins, label='after')
ax2.hist(kd.gnss.x-kd.groundtruth.interp_x(kd.gnss.time), bins=bins, label='before')

# e = np.array([kd.gnss.x-kd.groundtruth.interp_x(kd.gnss.time),kd.gnss.y-kd.groundtruth.interp_y(kd.gnss.time)]).T
# np.savetxt('kaist_urban17_gnss_errors.csv', e, delimiter=',')

ax1.legend()
ax2.legend()
ax1.grid()
ax2.grid()
plt.show()