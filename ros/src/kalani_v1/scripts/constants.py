class Constants:

    # dataset locations
    DATASET_DIRECTORY = '/home/pasan/kalani/data/2013-01-10'
    GNSS_DATA_PATH = DATASET_DIRECTORY + '/' + 'gps.csv'
    IMU_DATA_PATH = DATASET_DIRECTORY + '/' + 'ms25.csv'
    GROUNDTRUTH_DATA_PATH = DATASET_DIRECTORY + '/' + 'groundtruth.csv'

    # ros topics
    GNSS_DATA_TOPIC = 'gnss_data'
    IMU_DATA_TOPIC = 'imu_data'

    # ros node names
    FILTER_NODE_NAME = 'filter'
    GNSS_NODE_NAME = 'gnss'
    GNSS_LISTENER_NODE_NAME = 'gnss_listener'
    IMU_NODE_NAME = 'imu'
    IMU_LISTENER_NODE_NAME = 'imu_listener'