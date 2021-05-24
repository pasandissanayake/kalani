import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import struct
import cv2
import time
import datetime as dt

import pcl
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

from utilities import *

log = Log(prefix='kaist_datahandle')
config = get_config_dict()['kitti_dataset']

def extract_timestamps(file_path):
    timestamps  = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            cal = time.mktime(t.timetuple()) + t.microsecond * 1e-6
            timestamps.append(cal)
    return np.array(timestamps)

class Vector:
    def __init__(self):
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.z = np.zeros(0)


class GroundTruth:
    def __init__(self):
        self.time = np.zeros(2)
        self.x = np.zeros(2)
        self.y = np.zeros(2)
        self.z = np.zeros(2)
        self.r = np.zeros(2)
        self.p = np.zeros(2)
        self.h = np.zeros(2)
        self.interp_x = interp1d(self.time, self.x)
        self.interp_y = interp1d(self.time, self.y)
        self.interp_z = interp1d(self.time, self.z)
        self.interp_r = interp1d(self.time, self.r)
        self.interp_p = interp1d(self.time, self.p)
        self.interp_h = interp1d(self.time, self.h)


class GNSS:
    def __init__(self):
        self.time = np.zeros(0)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.covariance = np.zeros([4, 4])


class IMU:
    def __init__(self):
        self.time = np.zeros(0)
        self.acceleration = Vector()
        self.angular_rate = Vector()


class Altitude:
    def __init__(self):
        self.time = np.zeros(0)
        self.z = np.zeros(0)


class LaserScan:
    def __init__(self):
        self._directory = ''
        self._timestamp_file = ''
        self._file_list = []
        self.time = np.zeros(0)

    def set_directory(self, directory, timestamp_file):
        self._directory = directory
        self._file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        self._file_list.sort()
        self._timestamp_file = timestamp_file
        self.time = extract_timestamps(timestamp_file)

    def get_points(self, time, format='XYZI', nearesttimestamp=True):
            if nearesttimestamp:
                idx = np.argmin(np.abs(self.time - time))
            else:
                idx = np.where(self.time == time)[1][0]
            pointsfile = '{}/{}'.format(self._directory, self._file_list[idx])
            if os.path.isfile(pointsfile):
                scan = np.loadtxt(pointsfile, delimiter=' ', dtype=np.float32)
                w = scan.reshape((-1, 4))
                no_of_points = len(w)
                if format=='XYZI':
                    return no_of_points, w
                else:
                    return no_of_points, w[:, :3]
            else:
                log.log('LaserScan get_points() file not found. File name: {}'.format(pointsfile))
                return None

    def get_point_cloud2(self, header, format='XYZI', nearesttimestamp=True):
        c, p = self.get_points(header.stamp.to_sec(), format, nearesttimestamp)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, p)


class StereoImage:
    def __init__(self):
        self._left_dir = ''
        self._right_dir = ''
        self._timestamp_file = ''
        self._left_file_list=[]
        self._right_file_list=[]
        self.time = np.zeros(0)

    def set_directories(self, left_dir, right_dir, timestamps):
        self._left_dir = left_dir
        self._right_dir = right_dir
        self._timestamp_file = timestamps
        self._left_file_list = [f for f in os.listdir(left_dir) if os.path.isfile(os.path.join(left_dir, f))]
        self._left_file_list.sort()
        self._right_file_list = [f for f in os.listdir(right_dir) if os.path.isfile(os.path.join(right_dir, f))]
        self._right_file_list.sort()
        log.log("l:{}, r:{}".format(len(self._left_file_list), len(self._right_file_list)))
        self.time = extract_timestamps(timestamps)

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
        self.VEHICLE_R_VLP = np.eye(4)
        self.VEHICLE_R_STEREO = np.eye(4)


class KITTIData:

    def __init__(self):
        self.groundtruth = GroundTruth()
        self.gnss = GNSS()
        self.imu = IMU()
        self.altitude = Altitude()
        self.vlp = LaserScan()
        self.stereoImage = StereoImage()
        self.calibrations = Calibrations()

        self._GNSS_FLAG = False
        self._IMU_FLAG = False
        self._ALTITUDE_FLAG = False
        self._VLP_FLAG = False
        self._STEREO_IMAGE_FLAG = False

        self.GNSS_CLASS_NAME = self.gnss.__class__.__name__
        self.IMU_CLASS_NAME = self.imu.__class__.__name__
        self.ALTITUDE_CLASS_NAME = self.altitude.__class__.__name__
        self.VLP_CLASS_NAME = self.vlp.__class__.__name__
        self.STEREO_IMAGE_CLASS_NAME = self.stereoImage.__class__.__name__

    def load_groundtruth_from_numpy(self, timestamps, gt_array, origin):
        self.groundtruth.time = timestamps
        # set starting groundtruth point as (0, 0, 0)
        # groundtruth position is obtained from RTK GPS
        lat = gt_array[:, 0]
        lon = gt_array[:, 1]

        # convert latitudes and longitudes (in degrees) to ENU frame coordinates of the vehicle origin
        position = get_utm_from_gnss_fix(np.array([lat, lon]).T, origin[0:2], '32U', 'north', fixunit='deg')
        self.groundtruth.x = position[:, 0] - self.calibrations.VEHICLE_R_GNSS[0, 3]
        self.groundtruth.y = position[:, 1] - self.calibrations.VEHICLE_R_GNSS[1, 3]

        # obtain altitude from RTK GPS and set initial position to 0
        self.groundtruth.z = gt_array[:, 2] - origin[2] - self.calibrations.VEHICLE_R_GNSS[2, 3]

        eulers = gt_array[:, 3:6]
        # Groundtruth orientation is originally in NWU frame, with yaw=0 being the east direction
        # conversions for euler angles
        self.groundtruth.r = -eulers[:, 1]
        self.groundtruth.p = eulers[:, 0]
        self.groundtruth.h = eulers[:, 2]

        self.groundtruth.interp_x = interp1d(self.groundtruth.time, self.groundtruth.x, axis=0, bounds_error=False, fill_value=self.groundtruth.x[0], kind='linear')
        self.groundtruth.interp_y = interp1d(self.groundtruth.time, self.groundtruth.y, axis=0, bounds_error=False, fill_value=self.groundtruth.y[0], kind='linear')
        self.groundtruth.interp_z = interp1d(self.groundtruth.time, self.groundtruth.z, axis=0, bounds_error=False, fill_value=self.groundtruth.z[0], kind='linear')
        self.groundtruth.interp_r = interp1d(self.groundtruth.time, self.groundtruth.r, axis=0, bounds_error=False, fill_value=self.groundtruth.r[0], kind='linear')
        self.groundtruth.interp_p = interp1d(self.groundtruth.time, self.groundtruth.p, axis=0, bounds_error=False, fill_value=self.groundtruth.p[0], kind='linear')
        self.groundtruth.interp_h = interp1d(self.groundtruth.time, self.groundtruth.h, axis=0, bounds_error=False, fill_value=self.groundtruth.h[0], kind='linear')

    def load_gnss_from_numpy(self, timestamps, gnss_array, origin):
        self.gnss.time = timestamps

        lat = gnss_array[:, 0]
        lon = gnss_array[:, 1]

        # convert latitudes and longitudes (in degrees) to ENU frame coordinates of the vehicle origin
        # position = get_position_from_gnss_fix(np.array([lat, lon]).T, origin, fixunit='deg', originunit='deg')
        position = get_utm_from_gnss_fix(np.array([lat, lon]).T, origin, '32U', 'north', fixunit='deg')
        self.gnss.x = position[:, 0] - self.calibrations.VEHICLE_R_GNSS[0, 3]
        self.gnss.y = position[:, 1] - self.calibrations.VEHICLE_R_GNSS[1, 3]

        self.gnss.covariance = np.array([gnss_array[:, 4], gnss_array[:, 5], gnss_array[:, 7], gnss_array[:, 8]]).T

    def load_imu_from_numpy(self, timestamps, imu_array):
        self.imu.time = timestamps

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
        self.imu.angular_rate.x = -imu_array[:, 18]
        self.imu.angular_rate.y =  imu_array[:, 17]
        self.imu.angular_rate.z =  imu_array[:, 19]

        # imu accelerations (originally in ms-2, so nothing to convert)
        self.imu.acceleration.x = -imu_array[:, 12]
        self.imu.acceleration.y =  imu_array[:, 11]
        self.imu.acceleration.z =  imu_array[:, 13]

    def load_altitude_from_numpy(self, timestamps, altitude_array, origin):
        # altitude time conversion (ns to s)
        # altitude originally in meters, so no unit conversions
        self.altitude.time = timestamps
        self.altitude.z = altitude_array[:, 2] - origin[2]

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

        tv_R_vlp = np.zeros((4, 4))
        tv_R_vlp[0:3, 0:3] = np.reshape(sensor_calibrations['vehicle_R_vlp']['R'], (3, 3))
        tv_R_vlp[0:3, 3] = np.array(sensor_calibrations['vehicle_R_vlp']['T'])
        tv_R_vlp[3, 3] = 1
        ov_R_vlp = np.matmul(ov_R_tv, tv_R_vlp)
        self.calibrations.VEHICLE_R_VLP = ov_R_vlp

        tv_R_stereo = np.zeros((4, 4))
        tv_R_stereo[0:3, 0:3] = np.reshape(sensor_calibrations['vehicle_R_stereo']['R'], (3, 3))
        tv_R_stereo[0:3, 3] = np.array(sensor_calibrations['vehicle_R_stereo']['T'])
        tv_R_stereo[3, 3] = 1
        ov_R_stereo = np.matmul(ov_R_tv, tv_R_stereo)
        self.calibrations.VEHICLE_R_STEREO = ov_R_stereo

    def load_data(self, dataroot=None, date=None, drive=None, oxts=False, vlp=False, stereoimage=False, calibrations=False):
        kitti_config = get_config_dict()['kitti_dataset']
        
        if dataroot is None:
            dataroot = kitti_config['data_root']
        if date is None:
            date = kitti_config['date']
        if drive is None:
            drive = kitti_config['drive']

        sequence = 's{}_{}'.format(date, drive)

        if oxts:
            oxts_dir = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['dir_oxts_data'])
            timestamp_file = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['file_name_oxts_timestamps'])
            timestamps = extract_timestamps(timestamp_file)
            oxts_files = os.listdir(oxts_dir)
            oxts_files.sort()
            oxts_data = []
            for oxts_file in oxts_files:
                o = np.loadtxt('{}/{}'.format(oxts_dir, oxts_file), delimiter=' ')
                oxts_data.append(o)
            oxts_array = np.array(oxts_data)
            origin = np.array([kitti_config[sequence]['map_origin']['easting'], kitti_config[sequence]['map_origin']['northing'], kitti_config[sequence]['map_origin']['alt']])        
            self.load_imu_from_numpy(timestamps, oxts_array)
            self._IMU_FLAG = True
            self.load_gnss_from_numpy(timestamps, oxts_array, origin[0:2])
            self._GNSS_FLAG = True
            self.load_altitude_from_numpy(timestamps, oxts_array, origin)
            self._ALTITUDE_FLAG = True
            self.load_groundtruth_from_numpy(timestamps, oxts_array, origin)
        if vlp:
            directory = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['dir_lidar'])
            vlp_timestamp_file = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['file_name_lidar_timestamps'])
            self.vlp.set_directory(directory, vlp_timestamp_file)
            self._VLP_FLAG = True
        if stereoimage:
            l_dir = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['dir_left_camera'])
            r_dir = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['dir_right_camera'])
            stereo_timestamp_file = '{}/{}_drive_{}_extract/{}/{}'.format(dataroot, date, drive, date, kitti_config['file_name_camera_timestamps'])
            self.stereoImage.set_directories(l_dir, r_dir, stereo_timestamp_file)
            self._STEREO_IMAGE_FLAG = True
        if calibrations:
            self.load_calibrations_from_config(date)

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
        if self._VLP_FLAG:
            data_objects.append(self.vlp)
        if self._STEREO_IMAGE_FLAG:
            data_objects.append(self.stereoImage)
        return self.Player(data_objects, starttime=starttime)
