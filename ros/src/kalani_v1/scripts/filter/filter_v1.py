''' version 1.0.0 - modified to be compatible with python 2 '''

import numpy as np
from rotations_v1 import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


class Filter_V1():

    STATE_BUFFER_LENGTH = 5

    # Number of state variables excluding covariance matrix (P) and filter_time
    NO_STATE_VARIABLES = 5

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
        self.state_buffer = []

        #############################################################
        ################# Prediction step variables #################
        #############################################################

        # Prediction matrices
        self.Fi = np.zeros([15, 12])
        self.Fi[3:15, :] = np.eye(12)

        # Prediction variances
        self.var_imu_an = 0.0001
        self.var_imu_wn = 0.0005
        self.var_imu_aw = 0.0000
        self.var_imu_ww = 0.0000

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

    def initialize_state(self, p, v, q, g, ab, wb, P, startTime):
        self.p = p
        self.v = v
        self.q = q
        self.g = g
        self.ab = ab
        self.wb = wb

        self.P = P

        self.filter_time = startTime
        self.state_initialized = True

        self.store_state_in_buffer()

    def predict(self, am, wm, time, loadindex=-1):
        if not self.state_initialized:
            print('state not initialized')
            return

        self.load_state_from_buffer(loadindex)

        if self.filter_time > time:
            print('filter is ahead of time')
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
            self.q = Quaternion(self.q[0],self.q[1],self.q[2],self.q[3]).quat_mult_left(Quaternion(axis_angle=dt * (wm - self.wb)))
            self.ab = self.ab
            self.wb = self.wb

            self.filter_time = time

            if loadindex == -1:
                self.store_state_in_buffer()
                self.put_inputs_in_buffer(-1, am, wm)
            else:
                self.store_state_in_buffer(loadindex + 1)
                self.put_inputs_in_buffer(loadindex + 1, am, wm)

    def correct(self, y, time, sensor):
        index = min(range(len(self.state_buffer)), key=lambda i: abs(self.get_filter_time(i)-time))

        if time < self.get_filter_time(0):
            print('measurement is too old. measurement time:', time, 'filter time range:', self.get_filter_time(-1), ' to ', self.get_filter_time(0),)
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

        if sensor == Filter_V1.GNSS_WITH_ALT:
            V = np.diag([self.var_gnss_with_alt_horizontal, self.var_gnss_with_alt_horizontal, self.var_gnss_with_alt_vertical])
            H = self.Hx_gnss_with_alt.dot(X_dtheta)
            K = self.P.dot(H.T).dot(np.linalg.inv(H.dot(self.P).dot(H.T) + V))
            self.P = (np.eye(15) - K.dot(H)).dot(self.P)
            dx = K.dot(y - self.p)

        elif sensor == Filter_V1.GNSS_NO_ALT:
            V = np.diag([self.var_gnss_no_alt_horizontal,self.var_gnss_no_alt_horizontal])
            H = self.Hx_gnss_no_alt.dot(X_dtheta)
            K = self.P.dot(H.T).dot(np.linalg.inv(H.dot(self.P).dot(H.T) + V))
            self.P = (np.eye(15) - K.dot(H)).dot(self.P)
            dx = K.dot(y - np.array([self.p[0],self.p[1]]))

        elif sensor == Filter_V1.ODOM_WITH_ALT:
            V = np.diag([self.var_odom_with_alt, self.var_odom_with_alt, self.var_odom_with_alt])
            H = self.Hx_odom_with_alt.dot(X_dtheta)
            K = self.P.dot(H.T).dot(np.linalg.inv(H.dot(self.P).dot(H.T) + V))
            self.P = (np.eye(15) - K.dot(H)).dot(self.P)
            dx = K.dot(y - self.v)

        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = Quaternion(axis_angle=dx[6:9]).quat_mult_left(self.q)
        self.ab = self.ab + dx[9:12]
        self.wb = self.wb + dx[12:15]

        self.store_state_in_buffer(index)

        for i in range(index+1, len(self.state_buffer)):
            inputs = self.get_inputs_from_buffer(i)
            self.predict(inputs[0],inputs[1],self.get_filter_time(i),i-1)

    def get_state(self):
        return self.p, self.v, self.q, self.ab, self.wb, self.P

    def get_state_as_numpy(self):
        return np.array(np.concatenate(([self.filter_time],self.p,self.v,self.q,self.ab,self.wb))).flatten()

    def load_state_from_buffer(self, index=-1):
        self.p = self.state_buffer[index][0]
        self.v = self.state_buffer[index][1]
        self.q = self.state_buffer[index][2]
        self.ab = self.state_buffer[index][3]
        self.wb = self.state_buffer[index][4]
        self.P = self.state_buffer[index][Filter_V1.NO_STATE_VARIABLES + 0]
        self.filter_time = self.state_buffer[index][Filter_V1.NO_STATE_VARIABLES + 1]

    def store_state_in_buffer(self, index='append'):
        if index == 'append':
            self.state_buffer.append([self.p, self.v, self.q, self.ab, self.wb, self.P, self.filter_time, 'nan', 'nan'])
            if len(self.state_buffer) >= Filter_V1.STATE_BUFFER_LENGTH: self.state_buffer.pop(0)
        else:
            self.state_buffer[index][0:Filter_V1.NO_STATE_VARIABLES+2] = [self.p, self.v, self.q, self.ab, self.wb, self.P, self.filter_time]

    def get_filter_time(self, index):
        return self.state_buffer[index][Filter_V1.NO_STATE_VARIABLES+1]

    def get_inputs_from_buffer(self, index):
        return self.state_buffer[index][Filter_V1.NO_STATE_VARIABLES+2:Filter_V1.NO_STATE_VARIABLES+4]

    def put_inputs_in_buffer(self, index, am, wm):
        self.state_buffer[index][Filter_V1.NO_STATE_VARIABLES+2:Filter_V1.NO_STATE_VARIABLES+4] = [am, wm]

