import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import struct
import cv2

import pcl
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

from utilities import *

log = Log(prefix='kaist_datahandle') # initialize logging
config = get_config_dict()['kaist_dataset'] # fetch configuration data dictionary

# a class to hold 3-dimensional general vector
class Vector:
    def __init__(self):
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.z = np.zeros(0)


# class for ground truth
class GroundTruth:
    def __init__(self):
        self.time = np.zeros(2)
        self.x = np.zeros(2)    # x coordinate
        self.y = np.zeros(2)    # y coordinate
        self.z = np.zeros(2)    # z coordinate
        self.r = np.zeros(2)    # roll
        self.p = np.zeros(2)    # pitch
        self.h = np.zeros(2)    # yaw
        self.interp_x = interp1d(self.time, self.x) # interpolated x (along time axis)
        self.interp_y = interp1d(self.time, self.y) # interpolated y (along time axis)
        self.interp_z = interp1d(self.time, self.z) # interpolated z (along time axis)
        self.interp_r = interp1d(self.time, self.r) # interpolated roll (along time axis)
        self.interp_p = interp1d(self.time, self.p) # interpolated pitch (along time axis)
        self.interp_h = interp1d(self.time, self.h) # interpolated yaw (along time axis)


# class to hold GNSS data
class GNSS:
    def __init__(self):
        self.time = np.zeros(0)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.covariance = np.zeros([4, 4])


# class to hold IMU data
class IMU:
    def __init__(self):
        self.time = np.zeros(0)
        self.acceleration = Vector()
        self.angular_rate = Vector()
        self.magnetic_field = Vector()


# class to hold Altimeter data
class Altitude:
    def __init__(self):
        self.time = np.zeros(0)
        self.z = np.zeros(0)


# class to manipulate LiDAR data
class LaserScan:
    def __init__(self):
        self._directory = ''    # directory where the pointcloud files are
        self._file_list = []    # list of the names of the pointcloud files
        self.time = np.zeros(0) # list of timestamps

    # method to populate _directory, _file_list and time
    def set_directory(self, directory):
        self._directory = directory
        self._file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        self._file_list.sort() # sort file list alphabetically
        self.time = np.array([int(t[:-4]) / 1e9 for t in self._file_list])

    # method to read a pointcloud corresponding to a given timestamp
    # pcformat --> XYZI: (x, y, z, intensity) format, otherwise: (x, y, z) only
    # nearesttimestamp --> true: fetch the closest pointcloud file in time
    #                      false: fetch the pointcloud with the exact timestamp. If cannot be found, return None.     
    def get_points(self, time, pcformat='XYZI', nearesttimestamp=True):
            if nearesttimestamp:
                idx = np.argmin(np.abs(self.time - time)) # obtain the index of the nearest timestamp from self.time array
            else:
                idx = np.where(self.time == time)[1][0]
            pointsfile = '{}/{}'.format(self._directory, self._file_list[idx]) # generate file name based on timestamp
            if os.path.isfile(pointsfile):
                f = open(pointsfile, "rb") # open pointcloud file
                s = f.read() # read file completely into memory
                no_of_floats = len(s) / 4 # number of floats - 4 bytes per each float
                no_of_points = no_of_floats / 4 # number of points in the cloud - 4 floats for each point (x,y,z,intensity)
                w = np.reshape(struct.unpack('f' * no_of_floats, s), (-1,4)).astype(np.float32)
                if pcformat=='XYZI':
                    return no_of_points, w
                else:
                    return no_of_points, w[:, :3]
            else:
                log.log('LaserScan get_points() file not found. File name: {}'.format(pointsfile))
                return None

    # method to obtain a pointcloud message corresponding to a given timestamp
    def get_point_cloud2(self, header, pcformat='XYZI', nearesttimestamp=True):
        c, p = self.get_points(header.stamp.to_sec(), pcformat, nearesttimestamp)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, p)


# class to manipulate stereo camera images
class StereoImage:
    def __init__(self):
        self._left_dir = '' # directory path for left images
        self._right_dir = ''    # directory path for right images
        self._timestamp_file = ''   # path for the file containing timestamps
        self._left_file_list=[] # list of left images file names
        self._right_file_list=[] # list of right image file names
        self.time = np.zeros(0) # array to hold timestamps

    def set_directories(self, left_dir, right_dir, timestamps):
        self._left_dir = left_dir
        self._right_dir = right_dir
        self._timestamp_file = timestamps
        self._left_file_list = [f for f in os.listdir(left_dir) if os.path.isfile(os.path.join(left_dir, f))]
        self._left_file_list.sort()
        self._right_file_list = [f for f in os.listdir(right_dir) if os.path.isfile(os.path.join(right_dir, f))]
        self._right_file_list.sort()
        log.log("l:{}, r:{}".format(len(self._left_file_list), len(self._right_file_list)))
        self.time = np.loadtxt(timestamps, delimiter=',') * 1e-9

    def get_stereo_images(self, time, nearesttimestamp=True):
            if nearesttimestamp:
                idx = np.argmin(np.abs(self.time - time))
            else:
                idx = np.where(self.time == time)[1][0]
            leftfile = '{}/{}'.format(self._left_dir, self._left_file_list[idx])
            rightfile = '{}/{}'.format(self._right_dir, self._right_file_list[idx])
            if os.path.isfile(leftfile) and os.path.isfile(rightfile) :
                left_image = cv2.imread(leftfile, 0)
                right_image = cv2.imread(rightfile, 0)
                return left_image, right_image
            else:
                log.log('StereoImage get_stereo_images() files not found. File names: {}, {}'.format(leftfile, rightfile))
                return None, None
    

class Calibrations:
    def __init__(self):
        self.VEHICLE_R_IMU = np.eye(4)
        self.VEHICLE_R_GNSS = np.eye(4)
        self.VEHICLE_R_LEFTVLP = np.eye(4)
        self.VEHICLE_R_STEREO = np.eye(4)


class KAISTData:

    def __init__(self):
        self.groundtruth = GroundTruth()
        self.gnss = GNSS()
        self.imu = IMU()
        self.altitude = Altitude()
        self.vlpLeft = LaserScan()
        self.stereoImage = StereoImage()
        self.calibrations = Calibrations()

        self._GNSS_FLAG = False
        self._IMU_FLAG = False
        self._ALTITUDE_FLAG = False
        self._VLP_LEFT_FLAG = False
        self._STEREO_IMAGE_FLAG = False

        self.GNSS_CLASS_NAME = self.gnss.__class__.__name__
        self.IMU_CLASS_NAME = self.imu.__class__.__name__
        self.ALTITUDE_CLASS_NAME = self.altitude.__class__.__name__
        self.VLP_LEFT_CLASS_NAME = self.vlpLeft.__class__.__name__
        self.STEREO_IMAGE_CLASS_NAME = self.stereoImage.__class__.__name__

    def load_groundtruth_from_numpy(self, gt_array, origin):
        if np.ndim(gt_array) != 2:
            log.log('Groundtruth array dimension mismatch.')
            return 

        # groundtruth time from ns to s
        self.groundtruth.time = gt_array[:,0] / 1e9
        # set starting groundtruth point as (0, 0, 0)
        # groundtruth position is already in ENU frame
        # So, nothing to convert
        self.groundtruth.x = gt_array[:, 4] - origin[0]
        self.groundtruth.y = gt_array[:, 8] - origin[1]
        self.groundtruth.z = gt_array[:, 12] - origin[2]

        # convert rotation matrix to obtain groundtruth euler angles
        eulers = np.zeros([len(gt_array),3])
        for i in range(len(gt_array)):
            r_mat = np.array([gt_array[i,1:4], gt_array[i,5:8], gt_array[i,9:12]])
            eulers[i] = tft.euler_from_matrix(r_mat, axes='sxyz')
        # Groundtruth orientation is originally in NWU frame
        # conversions for euler angles
        self.groundtruth.r = -eulers[:, 1]
        self.groundtruth.p = eulers[:, 0]
        self.groundtruth.h = eulers[:, 2] - np.pi / 2

        self.groundtruth.interp_x = interp1d(self.groundtruth.time, self.groundtruth.x, axis=0, bounds_error=False, fill_value="extrapolate", kind='linear')
        self.groundtruth.interp_y = interp1d(self.groundtruth.time, self.groundtruth.y, axis=0, bounds_error=False, fill_value="extrapolate", kind='linear')
        self.groundtruth.interp_z = interp1d(self.groundtruth.time, self.groundtruth.z, axis=0, bounds_error=False, fill_value="extrapolate", kind='linear')
        self.groundtruth.interp_r = interp1d(self.groundtruth.time, self.groundtruth.r, axis=0, bounds_error=False, fill_value="extrapolate", kind='linear')
        self.groundtruth.interp_p = interp1d(self.groundtruth.time, self.groundtruth.p, axis=0, bounds_error=False, fill_value="extrapolate", kind='linear')
        self.groundtruth.interp_h = interp1d(self.groundtruth.time, self.groundtruth.h, axis=0, bounds_error=False, fill_value="extrapolate", kind='linear')

    def load_gnss_from_numpy(self, gnss_array, origin):
        if np.ndim(gnss_array) != 2:
            log.log('GNSS array dimension mismatch.')
            return

        # gnss timestamp conversion (ns to s)
        self.gnss.time = gnss_array[:, 0] / 1e9

        lat = gnss_array[:, 1]
        lon = gnss_array[:, 2]

        # convert latitudes and longitudes (in degrees) to ENU frame coordinates of the vehicle origin
        # position = get_position_from_gnss_fix(np.array([lat, lon]).T, origin, fixunit='deg', originunit='deg')
        position = get_utm_from_gnss_fix(np.array([lat, lon]).T, origin, '52S', 'north', fixunit='deg')
        self.gnss.x = position[:, 0] - self.calibrations.VEHICLE_R_GNSS[0, 3]
        self.gnss.y = position[:, 1] - self.calibrations.VEHICLE_R_GNSS[1, 3]

        self.gnss.covariance = np.array([gnss_array[:, 4], gnss_array[:, 5], gnss_array[:, 7], gnss_array[:, 8]]).T

    def load_imu_from_numpy(self, imu_array):
        if np.ndim(imu_array) != 2:
            log.log('IMU array dimension mismatch.')
            return

        # imu timestamp correction (ns to s)
        self.imu.time = imu_array[:, 0] / 1e9

        # dataset's imu frame:
        # x--> towards front of the vehicle
        # y--> towards left side of vehicle(when looking from behind)
        # z--> towards roof of the vehicle
        # our imu frame:
        # x--> towards right side of vehicle(when looking from behind)
        # y--> towards front of the vehicle
        # z--> towards roof of the vehicle
        # Hence,
        # our x = their -y
        # our y = their  x
        # our z = their  z

        # imu angular rates (originally in rad s-1, so nothing to convert)
        self.imu.angular_rate.x = -imu_array[:, 9]
        self.imu.angular_rate.y =  imu_array[:, 8]
        self.imu.angular_rate.z = imu_array[:, 10]

        # imu accelerations (originally in ms-2, so nothing to convert)
        self.imu.acceleration.x = -imu_array[:, 12]
        self.imu.acceleration.y =  imu_array[:, 11]
        self.imu.acceleration.z =  imu_array[:, 13]

        # imu magnetic field (original units: gauss, converted to Tesla)
        self.imu.magnetic_field.x = -imu_array[:, 15] * 1e-4
        self.imu.magnetic_field.y =  imu_array[:, 14] * 1e-4
        self.imu.magnetic_field.z =  imu_array[:, 16] * 1e-4

    def load_altitude_from_numpy(self, altitude_array, origin):
        if np.ndim(altitude_array) != 2:
            log.log('Altitude array dimension mismatch.')
            return

        # altitude time conversion (ns to s)
        # altitude originally in meters, so no unit conversions
        self.altitude.time = altitude_array[:, 0] / 1e9
        self.altitude.z = altitude_array[:, 1] - altitude_array[0, 1]

    @staticmethod
    def vector_nwu_to_enu(vector):
        if np.ndim(vector) == 1:
            R_nwu_enu = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            return np.array([-vector[1], vector[0], vector[2]])
        elif np.ndim(vector) == 2:
            return np.array([-vector[:, 1], vector[:, 0], vector[:, 2]]).T
        else:
            log.log('Vector/vector array dimension mismatch.')

    def load_calibrations_from_config(self, sequence):
        # calibration files specify vehicle_R_sensor transform matrices
        # tv: their vehicle frame
        # x--> towards front of the vehicle
        # y--> towards left side of vehicle(when looking from behind)
        # z--> towards roof of the vehicle
        # ov: our vehicle frame
        # x--> towards right side of the vehicle (when looking from behind)
        # y--> towards front of the vehicle
        # z--> towards roof of the vehicle

        ov_R_tv = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        sensor_calibrations = config['sensor_calibrations']

        tv_R_imu = np.zeros((4,4))
        tv_R_imu[0:3, 0:3] = np.reshape(sensor_calibrations['vehicle_R_imu']['R'], (3,3))
        tv_R_imu[0:3, 3] = np.array(sensor_calibrations['vehicle_R_imu']['T'])
        tv_R_imu[3,3] = 1
        ov_R_imu = np.matmul(ov_R_tv, tv_R_imu)
        self.calibrations.VEHICLE_R_IMU = ov_R_imu

        tv_R_gnss = np.zeros((4, 4))
        tv_R_gnss[0:3, 0:3] = np.reshape(sensor_calibrations['vehicle_R_gnss']['R'], (3, 3))
        tv_R_gnss[0:3, 3] = np.array(sensor_calibrations['vehicle_R_gnss']['T'])
        tv_R_gnss[3, 3] = 1
        ov_R_gnss = np.matmul(ov_R_tv, tv_R_gnss)
        self.calibrations.VEHICLE_R_GNSS = ov_R_gnss

        tv_R_leftvlp = np.zeros((4, 4))
        tv_R_leftvlp[0:3, 0:3] = np.reshape(sensor_calibrations['vehicle_R_leftvlp']['R'], (3, 3))
        tv_R_leftvlp[0:3, 3] = np.array(sensor_calibrations['vehicle_R_leftvlp']['T'])
        tv_R_leftvlp[3, 3] = 1
        ov_R_leftvlp = np.matmul(ov_R_tv, tv_R_leftvlp)
        self.calibrations.VEHICLE_R_LEFTVLP = ov_R_leftvlp

        tv_R_stereo = np.zeros((4, 4))
        tv_R_stereo[0:3, 0:3] = np.reshape(sensor_calibrations['vehicle_R_stereo']['R'], (3, 3))
        tv_R_stereo[0:3, 3] = np.array(sensor_calibrations['vehicle_R_stereo']['T'])
        tv_R_stereo[3, 3] = 1
        ov_R_stereo = np.matmul(ov_R_tv, tv_R_stereo)
        self.calibrations.VEHICLE_R_STEREO = ov_R_stereo

    def load_data(self, dataroot=None, sequence=None, groundtruth=False, imu=False, gnss=False, altitude=False, vlpleft=False, stereoimage=False, calibrations=False):
        kaist_config = get_config_dict()['kaist_dataset']

        if dataroot is None:
            dataroot = kaist_config['data_root']
        if sequence is None:
            sequence = kaist_config['sequence']

        groundtruth_file = '{}/{}/{}'.format(dataroot, sequence, kaist_config['file_name_groundtruth'])
        imu_file = '{}/{}/{}'.format(dataroot, sequence, kaist_config['file_name_imu'])
        gnss_file = '{}/{}/{}'.format(dataroot, sequence, kaist_config['file_name_gnss'])
        altimeter_file = '{}/{}/{}'.format(dataroot, sequence, kaist_config['file_name_altimeter'])

        if groundtruth:
            gt_array = np.loadtxt(groundtruth_file, delimiter=',')
            gt_array = gt_array[np.argsort(gt_array[:,0])]
            origin = np.array([kaist_config[sequence]['map_origin']['easting'], kaist_config[sequence]['map_origin']['northing'], kaist_config[sequence]['map_origin']['alt']])
            self.load_groundtruth_from_numpy(gt_array, origin)
        if imu:
            imu_array = np.loadtxt(imu_file, delimiter=',')
            imu_array = imu_array[np.argsort(imu_array[:,0])]
            self.load_imu_from_numpy(imu_array)
            self._IMU_FLAG = True
        if gnss:
            gnss_array = np.loadtxt(gnss_file, delimiter=',')
            gnss_array = gnss_array[np.argsort(gnss_array[:, 0])]
            origin = np.array([kaist_config[sequence]['map_origin']['easting'], kaist_config[sequence]['map_origin']['northing']])
            self.load_gnss_from_numpy(gnss_array, origin)
            self._GNSS_FLAG = True
        if altitude:
            altitude_array = np.loadtxt(altimeter_file, delimiter=',')
            altitude_array = altitude_array[np.argsort(altitude_array[:, 0])]
            origin = kaist_config[sequence]['map_origin']['alt']
            self.load_altitude_from_numpy(altitude_array, origin)
            self._ALTITUDE_FLAG = True
        if vlpleft:
            directory = '{}/{}/{}'.format(dataroot, sequence, kaist_config['dir_left_lidar'])
            self.vlpLeft.set_directory(directory)
            self._VLP_LEFT_FLAG = True
        if stereoimage:
            l_dir = '{}/{}/{}'.format(dataroot, sequence, kaist_config['dir_left_camera'])
            r_dir = '{}/{}/{}'.format(dataroot, sequence, kaist_config['dir_right_camera'])
            t_file = '{}/{}/{}'.format(dataroot, sequence, kaist_config['file_name_camera_timestamps'])
            self.stereoImage.set_directories(l_dir, r_dir, t_file)
            self._STEREO_IMAGE_FLAG = True
        if calibrations:
            self.load_calibrations_from_config(sequence)

    class Player:
        def __init__(self, data_objects, starttime=None):
            self._no_of_data_objects = len(data_objects)
            self._data_objects = data_objects
            self._time_stamp_indices = [0] * self._no_of_data_objects
            self._time_stamps = [-1] * self._no_of_data_objects
            self._data_object_flags = []
            for i in range(self._no_of_data_objects):
                if starttime is None:
                    self._time_stamps[i] = self._data_objects[i].time[0]
                else:
                    self._time_stamp_indices[i] = np.argmax(self._data_objects[i].time >= starttime)
                    self._time_stamps[i] = self._data_objects[i].time[self._time_stamp_indices[i]]
                self._data_object_flags.append(self._data_objects[i].__class__.__name__)

        def next(self):
            if len(self._data_objects) < 1:
                return None
            i = np.argmin(self._time_stamps)
            idx = self._time_stamp_indices[i]
            return_tup = (self._data_object_flags[i], idx, self._time_stamps[i])
            if idx == len(self._data_objects[i].time) - 1:
                self._data_objects.pop(i)
                self._data_object_flags.pop(i)
                self._time_stamp_indices.pop(i)
                self._time_stamps.pop(i)
            else:
                prev_time_stamp = self._time_stamps[i]
                while self._time_stamp_indices[i] < len(self._data_objects[i].time) and self._time_stamps[i] <= prev_time_stamp:
                    self._time_stamp_indices[i] += 1
                    self._time_stamps[i] = self._data_objects[i].time[self._time_stamp_indices[i]]
            return return_tup

    def get_player(self, starttime=None):
        data_objects = []
        if self._GNSS_FLAG:
            data_objects.append(self.gnss)
        if self._IMU_FLAG:
            data_objects.append(self.imu)
        if self._ALTITUDE_FLAG:
            data_objects.append(self.altitude)
        if self._VLP_LEFT_FLAG:
            data_objects.append(self.vlpLeft)
        if self._STEREO_IMAGE_FLAG:
            data_objects.append(self.stereoImage)
        return self.Player(data_objects, starttime=starttime)
