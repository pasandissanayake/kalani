class Constants:

    #####################################################
    ###################### Globals ######################
    #####################################################

    # ros topics
    STATE_TOPIC = 'state'
    ERROR_TOPIC = 'error'
    CONVERTED_GNSS_DATA_TOPIC = 'conv_data/gnss'
    CONVERTED_GROUNDTRUTH_DATA_TOPIC = 'conv_data/groundtruth'

    # ros node names
    LOCATOR_NODE_NAME = 'locator'
    EVALUATOR_NODE_NAME = 'evaluator'

    # frame ids
    WORLD_FRAME = 'world'       # The local ENU frame, stationary relative to the earth
    BODY_FRAME = 'body'         # Body frame of the vehicle. Denotes the estimate
    GROUNDTRUTH_FRAME = 'gt'    # Frame in which the ground truth is published
    GNSS_FRAME = 'gnss'         # Frame in which the converted GNSS location is published




    #####################################################
    ################### NCLT dataset ####################
    #####################################################

    # file locations
    NCLT_DATASET_DIRECTORY = '/home/entc/kalani/data/2012-02-02'

    NCLT_GNSS_DATA_FILE_NAME = 'gps.csv'
    NCLT_RTK_GNSS_DATA_FILE_NAME = 'gps_rtk.csv'
    NCLT_AHRS_DATA_FILE_NAME = 'ms25.csv'
    NCLT_GROUNDTRUTH_DATA_FILE_NAME = 'groundtruth.csv'
    NCLT_SENSOR_DATA_ROSBAG_FILE_NAME = 'sensor_data.bag'

    NCLT_GNSS_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + NCLT_GNSS_DATA_FILE_NAME
    NCLT_RTK_GNSS_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + NCLT_RTK_GNSS_DATA_FILE_NAME
    NCLT_AHRS_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + NCLT_AHRS_DATA_FILE_NAME
    NCLT_GROUNDTRUTH_DATA_PATH = NCLT_DATASET_DIRECTORY + '/' + NCLT_GROUNDTRUTH_DATA_FILE_NAME
    NCLT_SENSOR_DATA_ROSBAG_PATH = NCLT_DATASET_DIRECTORY + '/' + NCLT_SENSOR_DATA_ROSBAG_FILE_NAME


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
    NCLT_SENSOR_DATA_ROSBAG_NODE_NAME = 'nclt_rosbag'