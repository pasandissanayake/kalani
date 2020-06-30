''' version 1.0.0 - modified to be compatible with python 2 '''

import numpy as np
import threading
from rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


class Kalman_Filter_V1():

    STATE_BUFFER_LENGTH = 5

    # Number of state variables excluding covariance matrix (P) and filter_time
    NO_STATE_VARIABLES = 5

    # Allowed maximum gap between any two initial state variables (in seconds)
    STATE_INIT_TIME_THRESHOLD = 1

    # Correction identification constants
    GNSS_WITH_ALT = 1
    GNSS_NO_ALT = 2
    ODOM_WITH_ALT = 3


    def __init__(self):

        #############################################################
        ##################### State variables #######################
        #############################################################

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.zeros(4)
        self.ab = np.zeros(3)
        self.wb = np.zeros(3)
        self.g = np.zeros(3)
        self.P = np.zeros([15, 15])
        self.filter_time = 0
        self.state_initialized = False
        self.state_times = -1 * np.ones(Kalman_Filter_V1.NO_STATE_VARIABLES)
        self.state_buffer = []
        self._lock = threading.Lock()

        #############################################################
        ################# Prediction step variables #################
        #############################################################

        # Prediction matrices
        self.Fi = np.zeros([15, 12])
        self.Fi[3:15, :] = np.eye(12)

        # Prediction variances
        self.var_imu_an = 0.0001
        self.var_imu_wn = 0.0005
        self.var_imu_aw = 0.0001
        self.var_imu_ww = 0.0001

        #############################################################
        ################# Correction step variables #################
        #############################################################

        # Variables for GNSS with altitude
        self.var_gnss_with_alt_horizontal = 10.0
        self.var_gnss_with_alt_vertical = 200.0
        self.Hx_gnss_with_alt = np.zeros([3, 16])
        self.Hx_gnss_with_alt[:, 0:3] = np.eye(3)

        # Variables for GNSS without altitude
        self.var_gnss_no_alt_horizontal = self.var_gnss_with_alt_horizontal
        self.Hx_gnss_no_alt = np.zeros([2, 16])
        self.Hx_gnss_no_alt[:, 0:2] = np.eye(2)

        # Variables for velocity from odometry with altitude
        self.var_odom_with_alt = 10.0
        self.Hx_odom_with_alt = np.zeros([3,16])
        self.Hx_odom_with_alt[:, 3:6] = np.eye(3)


    def initialize_state(self, p=None, cov_p=None, v=None, cov_v=None, q=None, cov_q=None, ab=None, cov_ab=None, wb=None, cov_wb=None, g=None, time=-1):
        # with self._lock:
            if p is not None:
                if cov_p is not None:
                    self.p = p
                    self.P[0:3,0:3] = np.diag(cov_p)
                    self.state_times[0] = time
                else:
                    print('position covariances are not provided')

            if v is not None:
                if cov_v is not None:
                    self.v = v
                    self.P[3:6,3:6] = np.diag(cov_v)
                    self.state_times[1] = time
                else:
                    print('velocity covariances are not provided')

            if q is not None:
                if cov_q is not None:
                    self.q = q
                    self.P[6:9,6:9] = np.diag(cov_q)
                    self.state_times[2] = time
                else:
                    print('rotation covariances are not provided')

            if ab is not None:
                if cov_ab is not None:
                    self.ab = ab
                    self.P[9:12,9:12] = np.diag(cov_ab)
                    self.state_times[3] = time
                else:
                    print('acceleration bias covariances are not provided')

            if wb is not None:
                if cov_wb is not None:
                    self.wb = wb
                    self.P[12:15,12:15] = np.diag(cov_wb)
                    self.state_times[4] = time
                else:
                    print('angular velocity bias covariances are not provided')

            if g is not None:
                self.g = g

            self.filter_time = time

            initialized = all(st > 0 for st in self.state_times)
            if max(self.state_times)-min(self.state_times) < Kalman_Filter_V1.STATE_INIT_TIME_THRESHOLD and initialized:
                self.state_initialized = True
                self.store_state_in_buffer()
                print('state initialized')
            else:
                print('initial state time stamps:', self.state_times)


    def predict(self, am, wm, time, loadindex=-1, inputname='unspecified'):
        # with self._lock:
            if not self.state_initialized:
                print('state not initialized')
                return

            self.load_state_from_buffer(loadindex)

            if self.filter_time > time:
                print(inputname, 'input is too old. input time:', time, 'filter time:', self.filter_time)
                return

            else:
                dt = time - self.filter_time
                R_inert_body = Quaternion(self.q[0],self.q[1],self.q[2],self.q[3]).to_mat()

                Fx = np.eye(15)
                Fx[0:3, 3:6] = dt * np.eye(3)
                Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - self.ab)) * dt
                Fx[3:6, 9:12] = -dt * R_inert_body
                Fx[6:9, 12:15] = -dt * R_inert_body

                Qi = np.eye(12)
                Qi[0:3, 0:3] = self.var_imu_an * dt ** 2 * Qi[0:3, 0:3]
                Qi[3:6, 3:6] = self.var_imu_wn * dt ** 2 * Qi[3:6, 3:6]
                Qi[6:9, 6:9] = self.var_imu_aw * dt * Qi[6:9, 6:9]
                Qi[9:12, 9:12] = self.var_imu_ww * dt * Qi[9:12, 9:12]

                self.P = Fx.dot(self.P).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T)

                self.p = self.p + self.v * dt + 0.5 * (R_inert_body.dot(am - self.ab) + self.g) * dt ** 2
                self.v = self.v + (R_inert_body.dot(am - self.ab) + self.g) * dt
                self.q = Quaternion(self.q[0],self.q[1],self.q[2],self.q[3]).quat_mult_left(Quaternion(axis_angle=dt * (wm - self.wb)),out='Quaternion').normalize().to_numpy()
                self.ab = self.ab
                self.wb = self.wb

                self.filter_time = time

                if loadindex == -1:
                    self.store_state_in_buffer()
                    self.put_inputs_in_buffer(-1, am, wm)
                else:
                    self.store_state_in_buffer(loadindex + 1)
                    self.put_inputs_in_buffer(loadindex + 1, am, wm)


    def correct(self, y, Hx, V, time, measurementname='unspecified'):
        # with self._lock:
            index = self.get_nearest_index_by_time(time)

            if time < self.get_filter_time(0):
                print(measurementname, 'measurement is too old. measurement time:', time, 'filter time range:', self.get_filter_time(0), ' to ', self.get_filter_time(-1),)
                return

            self.load_state_from_buffer(index)

            dx = np.zeros(15)

            Q_dtheta = 0.5 * np.array([
                [-self.q[1], -self.q[2], -self.q[3]],
                [self.q[0], self.q[3], -self.q[2]],
                [-self.q[3], self.q[0], self.q[1]],
                [self.q[2], -self.q[1], self.q[0]]
            ])

            X_dtheta = np.zeros([16,15])
            X_dtheta[0:6,0:6] = np.eye(6)
            X_dtheta[6:10,6:9] = Q_dtheta
            X_dtheta[10:16,9:15] = np.eye(6)

            H = np.matmul(Hx,X_dtheta)
            K = np.matmul(np.matmul(self.P, H.T), np.linalg.inv(np.matmul(np.matmul(H, self.P), H.T) + V))
            self.P = np.matmul(np.eye(15) - np.matmul(K, H), self.P)
            state = self.get_state_as_numpy()[1:]
            dx = K.dot(y - np.matmul(Hx, state))

            self.p = self.p + dx[0:3]
            self.v = self.v + dx[3:6]
            self.q = Quaternion(axis_angle=dx[6:9]).quat_mult_left(self.q,out='Quaternion').normalize().to_numpy()
            self.ab = self.ab + dx[9:12]
            self.wb = self.wb + dx[12:15]

            if index<0 or index>=len(self.state_buffer):
                print('index out of range. index:', index, 'state buf length:', len(self.state_buffer))
            self.store_state_in_buffer(index)

            if index+1 < len(self.state_buffer):
                for i in range(index+1, len(self.state_buffer)):
                    inputs = self.get_inputs_from_buffer(i)
                    self.predict(inputs[0],inputs[1],self.get_filter_time(i),i-1,measurementname + '_correction')


    def get_state(self):
        return self.p, self.v, self.q, self.ab, self.wb, self.P


    def get_state_as_numpy(self):
        return np.array(np.concatenate(([self.filter_time],self.p,self.v,self.q,self.ab,self.wb))).flatten()


    def get_covariance_as_numpy(self):
        return self.P.flatten()


    def load_state_from_buffer(self, index=-1):
        self.p = self.state_buffer[index][0]
        self.v = self.state_buffer[index][1]
        self.q = self.state_buffer[index][2]
        self.ab = self.state_buffer[index][3]
        self.wb = self.state_buffer[index][4]
        self.P = self.state_buffer[index][Kalman_Filter_V1.NO_STATE_VARIABLES + 0]
        self.filter_time = self.state_buffer[index][Kalman_Filter_V1.NO_STATE_VARIABLES + 1]


    def store_state_in_buffer(self, index=None):
        # with self._lock:
            if index is None:
                self.state_buffer.append([self.p, self.v, self.q, self.ab, self.wb, self.P, self.filter_time, 'nan', 'nan'])
                if len(self.state_buffer) >= Kalman_Filter_V1.STATE_BUFFER_LENGTH: self.state_buffer.pop(0)
            else:
                self.state_buffer[index][0:Kalman_Filter_V1.NO_STATE_VARIABLES + 2] = [self.p, self.v, self.q, self.ab, self.wb, self.P, self.filter_time]


    def get_filter_time(self, index):
        return self.state_buffer[index][Kalman_Filter_V1.NO_STATE_VARIABLES + 1]


    def get_inputs_from_buffer(self, index):
        return self.state_buffer[index][Kalman_Filter_V1.NO_STATE_VARIABLES + 2:Kalman_Filter_V1.NO_STATE_VARIABLES + 4]


    def put_inputs_in_buffer(self, index, am, wm):
        # with self._lock:
            self.state_buffer[index][Kalman_Filter_V1.NO_STATE_VARIABLES + 2:Kalman_Filter_V1.NO_STATE_VARIABLES + 4] = [am, wm]


    def get_nearest_index_by_time(self,time):
        index = min(range(len(self.state_buffer)), key=lambda i: abs(self.get_filter_time(i) - time))
        return index
