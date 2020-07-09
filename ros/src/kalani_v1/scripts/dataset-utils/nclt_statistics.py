import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
sys.path.append('../')
from constants import Constants
from filter.rotations_v1 import Quaternion
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class GroundTruth:
    def __init__(self, length):
        self.time = np.zeros(length)

        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.z = np.zeros(length)

        self.r = np.zeros(length)
        self.p = np.zeros(length)
        self.h = np.zeros(length)

class ConvertedGNSS:
    def __init__(self, length):
        self.time = np.zeros(length)
        self.fix_mode = np.zeros(length)
        self.no_of_satellites = np.zeros(length)

        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.z = np.zeros(length)

        self.speed = np.zeros(length)
        self.track = np.zeros(length)


class NCLTData:

    def __init__(self):
        gt_array = np.loadtxt(Constants.NCLT_GROUNDTRUTH_DATA_PATH,delimiter=',')
        print('reading ground truth from file completed')
        self.groundtruth = GroundTruth(len(gt_array))
        self.groundtruth.time = gt_array[:,0] / 1e6
        self.groundtruth.x = gt_array[:,2]
        self.groundtruth.y = gt_array[:,1]
        self.groundtruth.z = -gt_array[:,3]
        self.groundtruth.r = gt_array[:,5]
        self.groundtruth.p = gt_array[:,4]
        self.groundtruth.r = -gt_array[:,6]
        print('ground truth conversion completed')

        gnss_array = np.loadtxt(Constants.NCLT_GNSS_DATA_PATH, delimiter=',')
        print('read gnss data from file')
        self.converted_gnss = ConvertedGNSS(len(gnss_array))
        self.converted_gnss.time = gnss_array[:,0] / 1e6
        self.converted_gnss.fix_mode = gnss_array[:,1]
        self.converted_gnss.no_of_satellites = gnss_array[:,2]

        origin = np.array([np.deg2rad(42.293227), np.deg2rad(-83.709657), 270])
        r = 6400000
        fix = gnss_array[:,3:6]
        dif = fix - origin

        self.converted_gnss.x = r * np.cos(origin[0]) * np.sin(dif[:, 1])
        self.converted_gnss.y = r * np.sin(dif[:,0])
        self.converted_gnss.z = dif[:,2]
        self.converted_gnss.speed = gnss_array[:,6]
        self.converted_gnss.track = gnss_array[:,7]
        print('gnss conversion completed')


    def get_groundtruth(self):
        return self.groundtruth


    def get_converted_gnss(self):
        return self.converted_gnss


dataset = NCLTData()
time = dataset.converted_gnss.time
gt_interpol_x = interp1d(dataset.groundtruth.time,dataset.groundtruth.x,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
error_x = dataset.converted_gnss.x - gt_interpol_x(time)
mean_x = np.average(error_x)
variance_x = np.average(error_x**2)
print 'var x:',variance_x, '\nmean x:',mean_x

gt_interpol_y = interp1d(dataset.groundtruth.time,dataset.groundtruth.y,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
error_y = dataset.converted_gnss.y - gt_interpol_y(time)
mean_y = np.average(error_y)
variance_y = np.average(error_y**2)
print 'var y:',variance_y, '\nmean y:',mean_y

gt_interpol_z = interp1d(dataset.groundtruth.time,dataset.groundtruth.z,axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
error_z = dataset.converted_gnss.z - gt_interpol_z(time)
mean_z = np.average(error_z)
variance_z = np.average(error_z**2)
print 'var z:',variance_z, '\nmean z:',mean_z

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
num_bins = 100
error_range = np.arange(-15,15,0.1)

ax1.hist(error_x, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax1.plot(error_range,mlab.normpdf(error_range, mean_x, np.sqrt(variance_x)))
ax1.set_title('x')

ax2.hist(error_y, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax2.plot(error_range,mlab.normpdf(error_range, mean_y, np.sqrt(variance_y)))
ax2.set_title('y')

ax3.hist(error_z, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax3.plot(error_range,mlab.normpdf(error_range, mean_z, np.sqrt(variance_z)))
ax3.set_title('z')

plt.show()

