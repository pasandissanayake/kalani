import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from utilities import *

log = Log(prefix='kaist_datahandle')
config = get_config_dict()['kaist_dataset']

class Vector:
    def __init__(self, length):
        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.z = np.zeros(length)


class GroundTruth:
    def __init__(self, length):
        self.length = length
        self.time = np.zeros(length)
        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.z = np.zeros(length)
        self.r = np.zeros(length)
        self.p = np.zeros(length)
        self.h = np.zeros(length)
        self.interp_x = None
        self.interp_y = None
        self.interp_z = None
        self.interp_r = None
        self.interp_p = None
        self.interp_h = None


class GNSS:
    def __init__(self, length):
        self.length = length
        self.time = np.zeros(length)
        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.covariance = np.zeros([length, 4])


class IMU:
    def __init__(self, length):
        self.length = length
        self.time = np.zeros(length)
        self.acceleration = Vector(length)
        self.angular_rate = Vector(length)
        self.magnetic_field = Vector(length)


class Altitude:
    def __init__(self, length):
        self.length = length
        self.time = np.zeros(length)
        self.z = np.zeros(length)


class Calibrations:
    def __init__(self):
        self.VEHICLE_R_IMU = np.eye(4)
        self.VEHICLE_R_GNSS = np.eye(4)
        self.VEHICLE_R_LEFTVLP = np.eye(4)


class KAISTData:

    def __init__(self):
        self.groundtruth = None
        self.gnss = None
        self.imu = None
        self.altitude = None
        self.calibrations = Calibrations()

    def load_groundtruth_from_numpy(self, gt_array):
        if np.ndim(gt_array) != 2:
            log.log('Groundtruth array dimension mismatch.')
            self.groundtruth = None

        groundtruth = GroundTruth(len(gt_array))

        # groundtruth time from ns to s
        groundtruth.time = gt_array[:,0] / 1e9
        # set starting groundtruth point as (0, 0, 0)
        # groundtruth position is already in ENU frame
        # So, nothing to convert
        groundtruth.x = gt_array[:, 4] - gt_array[0, 4]
        groundtruth.y = gt_array[:, 8] - gt_array[0, 8]
        groundtruth.z = gt_array[:, 12] - gt_array[0, 12]

        # convert rotation matrix to obtain groundtruth euler angles
        eulers = np.zeros([len(gt_array),3])
        for i in range(len(gt_array)):
            r_mat = np.array([gt_array[i,1:4], gt_array[i,5:8], gt_array[i,9:12]])
            eulers[i] = tft.euler_from_matrix(r_mat, axes='sxyz')
        # Groundtruth orientation is originally in NWU frame
        # conversions for euler angles
        groundtruth.r = -eulers[:, 1]
        groundtruth.p = eulers[:, 0]
        groundtruth.h = eulers[:, 2] - np.pi / 2

        groundtruth.interp_x = interp1d(groundtruth.time, groundtruth.x, axis=0, bounds_error=False, fill_value=groundtruth.x[0], kind='linear')
        groundtruth.interp_y = interp1d(groundtruth.time, groundtruth.y, axis=0, bounds_error=False, fill_value=groundtruth.y[0], kind='linear')
        groundtruth.interp_z = interp1d(groundtruth.time, groundtruth.z, axis=0, bounds_error=False, fill_value=groundtruth.z[0], kind='linear')
        groundtruth.interp_r = interp1d(groundtruth.time, groundtruth.r, axis=0, bounds_error=False, fill_value=groundtruth.r[0], kind='linear')
        groundtruth.interp_p = interp1d(groundtruth.time, groundtruth.p, axis=0, bounds_error=False, fill_value=groundtruth.p[0], kind='linear')
        groundtruth.interp_h = interp1d(groundtruth.time, groundtruth.h, axis=0, bounds_error=False, fill_value=groundtruth.h[0], kind='linear')

        self.groundtruth = groundtruth

    def load_gnss_from_numpy(self, gnss_array, origin):
        if np.ndim(gnss_array) != 2:
            log.log('GNSS array dimension mismatch.')
            self.gnss = None

        gnss = GNSS(len(gnss_array))
        # gnss timestamp conversion (ns to s)
        gnss.time = gnss_array[:, 0] / 1e9

        lat = gnss_array[:, 1]
        lon = gnss_array[:, 2]

        # convert lattitudes and longitudes (in degrees) to ENU frame coordinates
        position = get_position_from_gnss_fix(np.array([lat, lon]).T, origin, fixunit='deg', originunit='deg')
        gnss.x = position[:, 0]
        gnss.y = position[:, 1]

        gnss.covariance = np.array([gnss_array[:, 4], gnss_array[:, 5], gnss_array[:, 7], gnss_array[:, 8]]).T

        self.gnss = gnss

    def load_imu_from_numpy(self, imu_array):
        if np.ndim(imu_array) != 2:
            log.log('IMU array dimension mismatch.')
            self.imu = None

        imu = IMU(len(imu_array))

        # imu timestamp correction (ns to s)
        imu.time = imu_array[:, 0] / 1e9

        # imu frame:
        # x--> towards front of the vehicle
        # y--> towards left side of vehicle(when looking from behind)
        # z--> towards roof of the vehicle
        # However, no conversions are done here, as they will be carried on during the corrections using the sensor
        # calibration matrix

        # imu angular rates (originally in rad s-1, so nothing to convert)
        imu.angular_rate.x = imu_array[:, 8]
        imu.angular_rate.y = imu_array[:, 9]
        imu.angular_rate.z = imu_array[:, 10]

        # imu accelerations (originally in ms-2, so nothing to convert)
        imu.acceleration.x = imu_array[:, 11]
        imu.acceleration.y = imu_array[:, 12]
        imu.acceleration.z = imu_array[:, 13]

        # imu magnetic field (original units: gauss, converted to Tesla)
        imu.magnetic_field.x = imu_array[:, 14] * 1e-4
        imu.magnetic_field.y = imu_array[:, 15] * 1e-4
        imu.magnetic_field.z = imu_array[:, 16] * 1e-4

        self.imu = imu

    def load_altitude_from_numpy(self, altitude_array, origin):
        if np.ndim(altitude_array) != 2:
            log.log('Altitude array dimension mismatch.')
            self.altitude = None
        altitude = Altitude(len(altitude_array))
        # altitude time conversion (ns to s)
        # altitude originally in meters, so no unit conversions
        altitude.time = altitude_array[:, 0] / 1e9
        altitude.z = altitude_array[:, 1] - origin
        self.altitude = altitude

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

        sensor_calibrations = config[sequence]['sensor_calibrations']

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

    def load_data(self, data_root=None, sequence=None, groundtruth=True, imu=True, gnss=True, altitude=True):
        config = get_config_dict()['kaist_dataset']

        if data_root is None:
            data_root = config['data_root']
        if sequence is None:
            sequence = config['sequence']

        groundtruth_file = '{}/{}/{}'.format(data_root, sequence, config['file_name_groundtruth'])
        imu_file = '{}/{}/{}'.format(data_root, sequence, config['file_name_imu'])
        gnss_file = '{}/{}/{}'.format(data_root, sequence, config['file_name_gnss'])
        altimeter_file = '{}/{}/{}'.format(data_root, sequence, config['file_name_altimeter'])

        if groundtruth:
            gt_array = np.loadtxt(groundtruth_file, delimiter=',')
            self.load_groundtruth_from_numpy(gt_array)
        if imu:
            imu_array = np.loadtxt(imu_file, delimiter=',')
            self.load_imu_from_numpy(imu_array)
        if gnss:
            gnss_array = np.loadtxt(gnss_file, delimiter=',')
            origin = np.array([config[sequence]['map_origin_gnss_coordinates']['lat'], config[sequence]['map_origin_gnss_coordinates']['lon']])
            self.load_gnss_from_numpy(gnss_array, origin)
        if altitude:
            altitude_array = np.loadtxt(altimeter_file, delimiter=',')
            origin = config[sequence]['map_origin_gnss_coordinates']['alt']
            self.load_altitude_from_numpy(altitude_array, origin)


