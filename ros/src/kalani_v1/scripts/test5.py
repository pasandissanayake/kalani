from kitti_datahandle import KITTIData
from matplotlib import pyplot as plt

oxts_dir = "/home/entc/kalani-data/kitti/2011_10_03_drive_0027_extract/2011_10_03/oxts/data"
kd = KITTIData()
kd.load_data(oxts=True)

print kd.gnss.x[0], kd.gnss.y[0]
print kd.imu.acceleration.x[0], kd.imu.angular_rate.y[1]
print kd.groundtruth.z[0]

# plt.plot(kd.gnss.x, kd.gnss.y)
# plt.show()