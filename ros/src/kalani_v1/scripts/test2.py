import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from kaist_datahandle import *
import tf.transformations as tft

sw = Stopwatch()
sw.start()
kd = KAISTData()
kd.load_data(groundtruth=True, imu=False, gnss=True, altitude=True, vlpleft=False)
print "data loded.. time taken: {} s".format(sw.stop())

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(kd.gnss.time, kd.gnss.y, label='gnss')
ax1.plot(kd.groundtruth.time, kd.groundtruth.y, label='gt')
ax1.legend()
ax1.grid()

error = kd.groundtruth.interp_y(kd.gnss.time)-kd.gnss.y
ax2.hist(error, bins=20, label='error')
ax2.legend()
ax2.grid()


fig2, ax21 = plt.subplots(1,1)
ax21.plot(kd.gnss.x, kd.gnss.y, label='gnss')
ax21.plot(kd.groundtruth.x, kd.groundtruth.y, label='gt')
ax21.legend()
ax2.grid()


plt.show()

e_x = kd.groundtruth.interp_x(kd.gnss.time) - kd.gnss.x
m_x = np.average(e_x)
v_x = np.var(e_x)
e_y = kd.groundtruth.interp_y(kd.gnss.time) - kd.gnss.y
m_y = np.average(e_y)
v_y = np.var(e_y)
e_z = kd.groundtruth.interp_z(kd.altitude.time) - kd.altitude.z
m_z = np.average(e_z)
v_z = np.var(e_z)

print "m_x:", m_x, "  v_x:", v_x
print "m_y:", m_y, "  v_y:", v_y
print "m_z:", m_z, "  v_z:", v_z