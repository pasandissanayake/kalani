from kitti_datahandle import KITTIData
from matplotlib import pyplot as plt
from utilities import *

# oxts_dir = "/home/entc/kalani-data/kitti/2011_10_03_drive_0027_extract/2011_10_03/oxts/data"
# kd = KITTIData()
# kd.load_data(oxts=True)

# print kd.gnss.x[0], kd.gnss.y[0]
# print kd.imu.acceleration.x[0], kd.imu.angular_rate.y[1]
# print kd.groundtruth.z[0]

# plt.plot(kd.gnss.x, kd.gnss.y)
# plt.show()

# an, ax =  quaternion_to_angle_axis((0.13558924, -0.02549051, -0.00768248, -0.99040738))
# print an * ax
# print tft.quaternion_about_axis(an , ax)
# -0.03880773975165199, -0.005761202282615559, 0.041751843461783164

# easting: 316995.2876
# northing: 4155550.097
ds_config = get_config_dict()['kaist_dataset']
origin = np.array([ds_config[ds_config['sequence']]['map_origin']['easting'], ds_config[ds_config['sequence']]['map_origin']['northing'], ds_config[ds_config['sequence']]['map_origin']['alt']])
print origin
print get_gnss_fix_from_utm(np.zeros(2), origin[0:2], '52S', 'north', fixunit='deg')