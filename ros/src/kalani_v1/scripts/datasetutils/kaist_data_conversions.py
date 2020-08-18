import tf.transformations as tft
import numpy as np


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


class RawGNSS:
    def __init__(self, length):
        self.length = length
        self.time = np.zeros(length)
        self.fix_mode = np.zeros(length)
        self.no_of_satellites = np.zeros(length)

        self.lat = np.zeros(length)
        self.long = np.zeros(length)
        self.alt = np.zeros(length)

        self.covariance = np.zeros([length, 9])

        self.track = np.zeros(length)
        self.speed = np.zeros(length)


class ConvertedGNSS:
    def __init__(self, length):
        self.length = length
        self.time = np.zeros(length)
        self.fix_mode = np.zeros(length)
        self.no_of_satellites = np.zeros(length)

        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.z = np.zeros(length)

        self.covariance = np.zeros([length, 9])

        self.track = np.zeros(length)
        self.speed = np.zeros(length)


class KAISTDataConversions:

    @staticmethod
    def groundtruth_numpy_to_raw(gt_array):
        if np.ndim(gt_array) > 1:
            groundtruth = GroundTruth(len(gt_array))
            groundtruth.time = gt_array[:,0] / 1e9
            groundtruth.x = gt_array[:, 4] - 316001.6098
            groundtruth.y = gt_array[:, 8] - 4155469.996
            groundtruth.z = gt_array[:, 12] - 17.23766251

            eulers = np.zeros([len(gt_array),3])
            for i in range(len(gt_array)):
                r_mat = np.array([gt_array[i,1:4], gt_array[i,5:8], gt_array[i,9:12]])
                eulers[i] = tft.euler_from_matrix(r_mat, axes='sxyz')

            groundtruth.r = -eulers[:, 1]
            groundtruth.p = eulers[:, 0]
            groundtruth.h = eulers[:, 2] - np.pi / 2
        else:
            groundtruth = GroundTruth(1)
            groundtruth.time = gt_array[0] / 1e9
            groundtruth.x = gt_array[4] - 316001.6098
            groundtruth.y = gt_array[8] - 4155469.996
            groundtruth.z = gt_array[12] - 17.23766251

            r_mat = np.array([gt_array[1:4], gt_array[5:8], gt_array[9:12]])
            eulers = tft.euler_from_matrix(r_mat, axes='sxyz')
            groundtruth.r = -eulers[1]
            groundtruth.p = eulers[0]
            groundtruth.h = eulers[2] - np.pi / 2
        return groundtruth

    @staticmethod
    def groundtruth_raw_to_converted(raw_groundtruth):
        groundtruth = GroundTruth(raw_groundtruth.length)
        groundtruth.time = raw_groundtruth.time
        groundtruth.x = raw_groundtruth.x
        groundtruth.y = raw_groundtruth.y
        groundtruth.z = raw_groundtruth.z
        groundtruth.r = raw_groundtruth.r
        groundtruth.p = raw_groundtruth.p
        groundtruth.h = raw_groundtruth.h
        return groundtruth

    @staticmethod
    def groundtruth_numpy_to_converted(gt_array):
        raw_gt = KAISTDataConversions.groundtruth_numpy_to_raw(gt_array)
        return KAISTDataConversions.groundtruth_raw_to_converted(raw_gt)

    @staticmethod
    def gnss_numpy_to_raw(gnss_array, angle_unit='deg'):
        if np.ndim(gnss_array) > 1:
            gnss = RawGNSS(len(gnss_array))
            gnss.time = gnss_array[:, 0] / 1e9

            gnss.fix_mode = 2
            gnss.no_of_satellites = 4

            gnss.lat = gnss_array[:, 1]
            gnss.long = gnss_array[:, 2]
            gnss.alt = gnss_array[:, 3]
            if angle_unit=='rad':
                gnss.lat = np.rad2deg(gnss.lat)
                gnss.long = np.rad2deg(gnss.long)

            gnss.covariance = gnss_array[:, 4:13]
            gnss.track = 0
            gnss.speed = 0
        else:
            gnss = RawGNSS(1)
            gnss.time = gnss_array[0] / 1e9

            gnss.fix_mode = 2
            gnss.no_of_satellites = 4

            gnss.lat = gnss_array[1]
            gnss.long = gnss_array[2]
            gnss.alt = gnss_array[3]
            if angle_unit=='rad':
                gnss.lat = np.rad2deg(gnss.lat)
                gnss.long = np.rad2deg(gnss.long)

            gnss.covariance = gnss_array[4:13]
            gnss.track = 0
            gnss.speed = 0
        return gnss

    @staticmethod
    def gnss_raw_to_converted(raw_gnss):
        converted_gnss = ConvertedGNSS(raw_gnss.length)
        converted_gnss.time = raw_gnss.time
        converted_gnss.fix_mode = raw_gnss.fix_mode
        converted_gnss.no_of_satellites = raw_gnss.no_of_satellites

        origin = np.array([np.deg2rad(37.527883), np.deg2rad(126.9176492), 37.26])
        r = 6400000
        fix = [raw_gnss.lat, raw_gnss.long, raw_gnss.alt]
        fix[0:2] = np.deg2rad(fix[0:2])
        dif = np.array([fix[0]-origin[0], fix[1]-origin[1], fix[2]-origin[2]])
        converted_gnss.x = r * np.cos(origin[0]) * np.sin(dif[1])
        converted_gnss.y = r * np.sin(dif[0])
        converted_gnss.z = dif[2]

        converted_gnss.track = raw_gnss.track
        converted_gnss.speed = raw_gnss.speed

        return converted_gnss

    @staticmethod
    def gnss_numpy_to_converted(gnss_array, angle_unit='deg'):
        gnss_raw = KAISTDataConversions.gnss_numpy_to_raw(gnss_array, angle_unit)
        return KAISTDataConversions.gnss_raw_to_converted(gnss_raw)

    @staticmethod
    def vector_nwu_to_enu(vector):
        R_nwu_enu = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # R_nwu_enu = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return np.matmul(R_nwu_enu, vector)


class KAISTData:

    def __init__(self, data_directory, gt=True, gnss=True):
        if gt:
            gt_array = np.loadtxt(data_directory + '/' + 'global_pose.csv', delimiter=',')
            self.groundtruth = KAISTDataConversions.groundtruth_numpy_to_converted(gt_array)

        if gnss:
            gnss_array = np.loadtxt(data_directory + '/' + 'sensor_data/gps.csv', delimiter=',')
            self.converted_gnss = KAISTDataConversions.gnss_numpy_to_converted(gnss_array, 'deg')

