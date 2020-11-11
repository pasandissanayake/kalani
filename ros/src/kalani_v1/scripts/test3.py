#!/usr/bin/env python2

import numpy as np
from scipy.interpolate import interp1d
from datasetutils.nclt_data_conversions import *
from matplotlib import pyplot as plt
from constants import Constants
from kaist_datahandle import *

class Foo:
    def __init__(self, states):
        for state in states:
            self.__dict__[state[0]] = np.zeros(state[1])

states = [
    ['a', 3],
    ['b', 4]
]
foo = Foo(states)
foo.a = np.array((2,1,3))
print foo.a, foo.b

nds = NCLTData(Constants.NCLT_DATASET_DIRECTORY)
a = np.array([nds.groundtruth.x, nds.groundtruth.y]).T
ngt = interp1d(nds.groundtruth.time, a, axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')

ner = np.array([ngt(nds.converted_gnss.time)[:,0] - nds.converted_gnss.x, ngt(nds.converted_gnss.time)[:,1] - nds.converted_gnss.y]).T


kds = KAISTData()
kds.load_data(groundtruth=True, imu=False, gnss=True, altitude=False, vlpleft=False)

ker = np.array([kds.groundtruth.interp_x(kds.gnss.time)-kds.gnss.x, kds.groundtruth.interp_y(kds.gnss.time)-kds.gnss.y]).T


# GNSS error first difference plots

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(nds.converted_gnss.time[:-1]-nds.converted_gnss.time[0], np.diff(ner[:, 0]))
ax1.legend()
ax1.grid()



## GNSS error plots
#
# fig, (ax1, ax2) = plt.subplots(2,1)
#
# ax1.plot(nds.converted_gnss.time-nds.converted_gnss.time[0], ner[:,0], label='x-error-nclt')
# ax1.plot(kds.gnss.time-kds.gnss.time[0], ker[:,0], label='x-error-kaist')
# ax1.plot(kds.gnss.time-kds.gnss.time[0], np.zeros(len(ker[:,0])))
# ax1.legend()
# ax1.grid()
#
# ax2.plot(nds.converted_gnss.time-nds.converted_gnss.time[0], ner[:,1], label='y-error-nclt')
# ax2.plot(kds.gnss.time-kds.gnss.time[0], ker[:,1], label='y-error-kaist')
# ax2.plot(kds.gnss.time-kds.gnss.time[0], np.zeros(len(ker[:,0])))
# ax2.legend()
# ax2.grid()

plt.show()