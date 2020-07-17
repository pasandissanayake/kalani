import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from nclt_data_conversions import *
sys.path.append('../')
from constants import Constants


dataset = NCLTData(Constants.NCLT_DATASET_DIRECTORY)
# dataset = NCLTData('/home/entc/kalani/data/2013-01-10')

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
error_z = dataset.converted_gnss.z[dataset.converted_gnss.fix_mode==3] - gt_interpol_z(time[dataset.converted_gnss.fix_mode==3])
mean_z = np.average(error_z)
variance_z = np.average(error_z**2)
print 'var z:',variance_z, '\nmean z:',mean_z


fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# error distributions
num_bins = 100
error_range = np.arange(-15,15,0.1)

ax1.hist(error_x, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax1.plot(error_range,mlab.normpdf(error_range, mean_x, np.sqrt(variance_x)))
ax1.set_title('x')

ax2.hist(error_y, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax2.plot(error_range,mlab.normpdf(error_range, mean_y, np.sqrt(variance_y)))
ax2.set_title('y')

z_error_range = np.arange(-200,200,1)
ax3.hist(error_z, num_bins, facecolor='blue', alpha=0.5, normed=True)
ax3.plot(z_error_range,mlab.normpdf(z_error_range, mean_z, np.sqrt(variance_z)))
ax3.set_title('z')

# ax1.plot(time, error_x)
# ax2.plot(time, error_y)

plt.show()

