class Constants:

    #####################################################
    ###################### Globals ######################
    #####################################################

    # file locations
    DATASET_DIRECTORY = '~/kalani/data/2012-04-29'
    GNSS_DATA_PATH = DATASET_DIRECTORY + '/' + 'gps.csv'
    IMU_DATA_PATH = DATASET_DIRECTORY + '/' + 'ms25.csv'
    GROUNDTRUTH_DATA_PATH = DATASET_DIRECTORY + '/' + 'groundtruth.csv'

    # ros topics
    ERROR_TOPIC = 'error'
    GNSS_DATA_TOPIC = 'gnss_data'
    GROUNDTRUTH_DATA_TOPIC = 'groundtruth_data'
    IMU_DATA_TOPIC = 'imu_data'
    STATE_TOPIC = 'state'

    # ros node names
    FILTER_NODE_NAME = 'filter'
    ERROR_CAL_NODE_NAME = 'errorcal'

    # frame ids
    GNSS_FRAME = 'gnss_frame'
    IMU_FRAME = 'imu_frame'
    STATE_FRAME = 'state_frame'




    #####################################################
    ################### NCLT dataset ####################
    #####################################################

    # file locations
    NCLT_SENSOR_DATA_ROSBAG_PATH = DATASET_DIRECTORY + '/' + 'sensor_data.bag'

    # raw data topics
    NCLT_RAW_DATA_GNSS_FIX_TOPIC = 'raw_data/gnss_fix'
    NCLT_RAW_DATA_GNSS_SPEED_TOPIC = 'raw_data/gnss_speed'
    NCLT_RAW_DATA_GNSS_TRACK_TOPIC = 'raw_data/gnss_track'

    NCLT_RAW_DATA_RTK_GNSS_FIX_TOPIC = 'raw_data/rtk_gnss_fix'
    NCLT_RAW_DATA_RTK_GNSS_SPEED_TOPIC = 'raw_data/rtk_gnss_speed'
    NCLT_RAW_DATA_RTK_GNSS_TRACK_TOPIC = 'raw_data/rtk_gnss_track'

    NCLT_RAW_DATA_IMU_TOPIC = 'raw_data/imu'
    NCLT_RAW_DATA_MAGNETOMETER_TOPIC = 'raw_data/magnetometer'

    # ros node names
    NCLT_GNSS_NODE_NAME = 'nclt_gnss'
    NCLT_GROUNDTRUTH_NODE_NAME = 'nclt_groundtruth'
    NCLT_IMU_NODE_NAME = 'nclt_imu'
    NCLT_SENSOR_DATA_ROSBAG_NODE_NAME = 'nclt_rosbag'