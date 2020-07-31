import os
import rospy
from definitions import ROOT_DIR


class BaseRawData:
    """Base class to initialize the directories for the raw data

    USAGE:
            BaseRawData('2013-01-10')

    """
    def __init__(self, date):

        self.date = date

        RAW_DATA_PATH_DFLT = '/home/entc/kalani/data/nclt'
        self.raw_data_path = rospy.get_param('~raw_data_path', RAW_DATA_PATH_DFLT)

        # init raw directory
        if self.raw_data_path.endswith('/'):
            self.raw_data_dir = self.raw_data_path + str(self.date)
        else:
            self.raw_data_dir = self.raw_data_path + '/' + str(self.date)

        # check if data exists
        if os.path.exists(self.raw_data_dir):

            self.ground_truth_dir = self.raw_data_dir
            if os.path.exists(self.ground_truth_dir):
                self.ground_truth_flag = True
            else:
                self.ground_truth_flag = False

            self.ground_truth_covariance_dir = self.raw_data_dir
            if os.path.exists(self.ground_truth_covariance_dir):
                self.ground_truth_covariance_flag = True
            else:
                self.ground_truth_covariance_flag = False

            self.hokuyo_data_dir = self.raw_data_dir + '/empty'
            if os.path.exists(self.hokuyo_data_dir):
                self.hokuyo_data_flag = True
            else:
                self.hokuyo_data_flag = False

            self.sensor_data_dir = self.raw_data_dir
            if os.path.exists(self.sensor_data_dir):
                self.sensor_data_flag = True
            else:
                self.sensor_data_flag = False

            self.velodyne_data_dir = self.raw_data_dir
            if os.path.exists(self.velodyne_data_dir):
                self.velodyne_data_flag = True
                self.velodyne_sync_data_dir = self.raw_data_dir + '/velodyne_sync/'
            else:
                self.velodyne_data_flag = False

            self.images_dir = self.raw_data_dir + '/empty'
            if os.path.exists(self.images_dir):
                self.images_flag = True
                self.images_lb3_dir = self.raw_data_dir + '/lb3/'
            else:
                self.images_flag = False

        else:
            raise ValueError("raw_data directory not exists")

