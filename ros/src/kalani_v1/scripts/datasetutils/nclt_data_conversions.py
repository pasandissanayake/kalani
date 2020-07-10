import numpy as np
import sys
sys.path.append('../')
from constants import Constants


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

        self.track = np.zeros(length)
        self.speed = np.zeros(length)


class NCLTDataConversions:

    @staticmethod
    def groundtruth_numpy_to_raw(gt_array):
        if np.ndim(gt_array) > 1:
            groundtruth = GroundTruth(len(gt_array))
            groundtruth.time = gt_array[:,0] / 1e6
            groundtruth.x = gt_array[:, 1]
            groundtruth.y = gt_array[:, 2]
            groundtruth.z = gt_array[:, 3]
            groundtruth.r = gt_array[:, 4]
            groundtruth.p = gt_array[:, 5]
            groundtruth.h = gt_array[:, 6]
        else:
            groundtruth = GroundTruth(1)
            groundtruth.time = gt_array[0] / 1e6
            groundtruth.x = gt_array[1]
            groundtruth.y = gt_array[2]
            groundtruth.z = gt_array[3]
            groundtruth.r = gt_array[4]
            groundtruth.p = gt_array[5]
            groundtruth.h = gt_array[6]
        return groundtruth

    @staticmethod
    def groundtruth_raw_to_converted(raw_groundtruth):
        groundtruth = GroundTruth(raw_groundtruth.length)
        groundtruth.time = raw_groundtruth.time
        groundtruth.x = raw_groundtruth.y
        groundtruth.y = raw_groundtruth.x
        groundtruth.z = -raw_groundtruth.z
        groundtruth.r = raw_groundtruth.p
        groundtruth.p = raw_groundtruth.r
        groundtruth.h = -raw_groundtruth.h
        return groundtruth

    @staticmethod
    def grountruthu_numpy_to_converted(gt_array):
        raw_gt = NCLTDataConversions.groundtruth_numpy_to_raw(gt_array)
        return NCLTDataConversions.groundtruth_raw_to_converted(raw_gt)

    @staticmethod
    def gnss_numpy_to_raw(gnss_array, angle_unit='deg'):
        if np.ndim(gnss_array) > 1:
            gnss = RawGNSS(len(gnss_array))
            gnss.time = gnss_array[:, 0] / 1e6

            gnss.fix_mode = gnss_array[:, 1]
            gnss.fix_mode = np.where(~np.isnan(gnss_array[:, 3]) & ~np.isnan(gnss_array[:, 4]) & ~np.isnan(gnss_array[:, 5]), 3, gnss.fix_mode)
            gnss.fix_mode = np.where(~np.isnan(gnss_array[:, 3]) & ~np.isnan(gnss_array[:, 4]) & np.isnan(gnss_array[:, 5]), 2, gnss.fix_mode)
            gnss.fix_mode = np.where(np.isnan(gnss_array[:, 3]) | np.isnan(gnss_array[:, 4]), 1, gnss.fix_mode)

            gnss.no_of_satellites = gnss_array[:, 2]

            gnss.lat = gnss_array[:, 3]
            gnss.long = gnss_array[:, 4]
            gnss.alt = gnss_array[:, 5]
            if angle_unit=='rad':
                gnss.lat = np.rad2deg(gnss.lat)
                gnss.long = np.rad2deg(gnss.long)

            gnss.track = gnss_array[:, 6]
            gnss.speed = gnss_array[:, 7]
        else:
            gnss = RawGNSS(1)
            gnss.time = gnss_array[0] / 1e6

            gnss.fix_mode = gnss_array[1]
            gnss.fix_mode = np.where(
                ~np.isnan(gnss_array[3]) & ~np.isnan(gnss_array[4]) & ~np.isnan(gnss_array[5]), 3,
                gnss.fix_mode)
            gnss.fix_mode = np.where(
                ~np.isnan(gnss_array[3]) & ~np.isnan(gnss_array[4]) & np.isnan(gnss_array[5]), 2,
                gnss.fix_mode)
            gnss.fix_mode = np.where(np.isnan(gnss_array[3]) | np.isnan(gnss_array[4]), 1, gnss.fix_mode)

            gnss.no_of_satellites = gnss_array[2]

            gnss.lat = gnss_array[3]
            gnss.long = gnss_array[4]
            gnss.alt = gnss_array[5]
            if angle_unit=='rad':
                gnss.lat = np.rad2deg(gnss.lat)
                gnss.long = np.rad2deg(gnss.long)

            gnss.track = gnss_array[6]
            gnss.speed = gnss_array[7]
        return gnss

    @staticmethod
    def gnss_raw_to_converted(raw_gnss):
        converted_gnss = ConvertedGNSS(raw_gnss.length)
        converted_gnss.time = raw_gnss.time
        converted_gnss.fix_mode = raw_gnss.fix_mode
        converted_gnss.no_of_satellites = raw_gnss.no_of_satellites

        origin = np.array([np.deg2rad(42.293227), np.deg2rad(-83.709657), 270])
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
        gnss_raw = NCLTDataConversions.gnss_numpy_to_raw(gnss_array, angle_unit)
        return NCLTDataConversions.gnss_raw_to_converted(gnss_raw)

    @staticmethod
    def vector_ned_to_enu(vector):
        R_ned_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        return np.matmul(R_ned_enu, vector)


class NCLTData:

    def __init__(self, data_directory):
        gt_array = np.loadtxt(data_directory + '/' + Constants.NCLT_GROUNDTRUTH_DATA_FILE_NAME, delimiter=',')
        self.groundtruth = NCLTDataConversions.grountruthu_numpy_to_converted(gt_array)

        gnss_array = np.loadtxt(data_directory + '/' + Constants.NCLT_GNSS_DATA_FILE_NAME, delimiter=',')
        self.converted_gnss = NCLTDataConversions.gnss_numpy_to_converted(gnss_array, 'rad')

