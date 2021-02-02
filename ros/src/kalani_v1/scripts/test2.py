import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from kaist_datahandle import *
import tf.transformations as tft

kd = KAISTData()
kd.load_data(sequence='urban27', imu=False, gnss=False, vlpleft=False, calibrations=True)

plt.plot(kd.altitude.time, kd.altitude.z, label='al')
plt.plot(kd.groundtruth.time, kd.groundtruth.z, label='gt')
plt.legend()
plt.show()

print np.average(kd.altitude.z - kd.groundtruth.interp_z(kd.altitude.time))
print kd.calibrations.VEHICLE_R_LEFTVLP