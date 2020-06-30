class Constants:

    #####################################################
    ###################### Globals ######################
    #####################################################

    # ros topics
    ERROR_TOPIC = 'error'
    GNSS_DATA_TOPIC = 'gnss_data'
    GROUNDTRUTH_DATA_TOPIC = 'groundtruth_data'
    IMU_DATA_TOPIC = 'imu_data'
    STATE_TOPIC = 'state'

    # ros node names
    LOCATOR_NODE_NAME = 'locator'
    EVALUATOR_NODE_NAME = 'evaluator'

    # frame ids
    GNSS_FRAME = 'gnss_frame'
    IMU_FRAME = 'imu_frame'
    STATE_FRAME = 'state_frame'




    #####################################################
    ################### NCLT dataset ####################
    #####################################################

    # file locations
    NCLT_DATASET_DIRECTORY = '/home/entc/kalani/data/2012-04-29'
    NCLT_GNSS_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + 'gps.csv'
    NCLT_RTK_GNSS_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + 'gps_rtk.csv'
    NCLT_AHRS_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + 'ms25.csv'
    NCLT_GROUNDTRUTH_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + 'groundtruth.csv'
    NCLT_SENSOR_DATA_ROSBAG_PATH = NCLT_DATASET_DIRECTORY + '/' + 'sensor_data.bag'


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