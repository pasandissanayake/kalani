import numpy as np
import threading
from copy import deepcopy

import tf.transformations as tft

from kalman_buffer import KalmanBuffer, KalmanStateObject
from utilities import *


log = Log('kalman_filter')


class KalmanFilter():
    def __init__(self):
        config = get_config_dict()['kalman_filter']

        # State buffer length
        self.STATE_BUFFER_LENGTH = config['buffer_size']

        # Allowed maximum gap between any two initial state variables (in seconds)
        self.STATE_INIT_TIME_THRESHOLD = config['state_init_time_threshold']

        self._g = np.array([0, 0, config['gravity']])

        self._state_buffer = KalmanBuffer(self.STATE_BUFFER_LENGTH)
        self._lock = threading.RLock()

        # Prediction matrices
        self.Fi = np.zeros([15, 12])
        self.Fi[3:15, :] = np.eye(12)

        # Bias variances
        self._aw_var = np.ones(3) * 1e-10
        self._ww_var = np.ones(3) * 1e-10


    def initialize_state(self, p=None, cov_p=None, v=None, cov_v=None, q=None, cov_q=None, ab=None, cov_ab=None, wb=None, cov_wb=None, time=-1):
        with self._lock:
            if self._state_buffer.get_buffer_length() < 1: self._state_buffer.add_state()

            st = self._state_buffer.get_state(-1)
            if p is not None:
                if cov_p is not None:
                    st.position.value = np.array(p)
                    st.position.time = time
                    st.covariance[0:3,0:3] = np.diag(cov_p)
                    self._state_buffer.update_state(st,-1)
                else:
                    log.log('position covariances are not provided')

            if v is not None:
                if cov_v is not None:
                    st.velocity.value = np.array(v)
                    st.velocity.time = time
                    st.covariance[3:6, 3:6] = np.diag(cov_v)
                    self._state_buffer.update_state(st, -1)
                else:
                    log.log('velocity covariances are not provided')

            if q is not None:
                if cov_q is not None:
                    st.orientation.value = np.array(q)
                    st.orientation.time = time
                    st.covariance[6:9, 6:9] = np.diag(cov_q)
                    self._state_buffer.update_state(st, -1)
                else:
                    log.log('rotation covariances are not provided')

            if ab is not None:
                if cov_ab is not None:
                    st.accel_bias.value = np.array(ab)
                    st.accel_bias.time = time
                    st.covariance[9:12, 9:12] = np.diag(cov_ab)
                    self._state_buffer.update_state(st, -1)
                    self._aw_var = np.array(cov_ab)
                else:
                    log.log('acceleration bias covariances are not provided')

            if wb is not None:
                if cov_wb is not None:
                    st.angular_bias.value = np.array(wb)
                    st.angular_bias.time = time
                    st.covariance[0:3, 0:3] = np.diag(cov_wb)
                    self._state_buffer.update_state(st, -1)
                    self._ww_var = np.array(cov_wb)
                else:
                    log.log('angular velocity bias covariances are not provided')

            st = self._state_buffer.get_state(-1)
            timestamps = st.times_to_numpy()
            initialized = all(ts >= 0 for ts in timestamps)
            if self._state_buffer.get_timestamp_of_newest_variable(-1) - self._state_buffer.get_timestamp_of_oldest_variable(-1) < self.STATE_INIT_TIME_THRESHOLD and initialized:
                st.state_time = time
                st.initialized = True
                self._state_buffer.update_state(st, -1)
                log.log('state initialized')
            else:
                log.log('initial state time stamps:', timestamps)


    def predict(self, am, var_am, wm, var_wm, time, loadindex=-1, inputname='unspecified'):
        with self._lock:
            st = self._state_buffer.get_state(loadindex)
            if st is None or not st.initialized:
                log.log('state not initialized')
                return

            st_time = st.state_time
            if st_time >= time:
                log.log(inputname, 'input is too old. input time:', time, 'filter time:', st_time, 'load index:', loadindex)
                return

            else:
                dt = time - st_time

                p = st.position.value
                v = st.velocity.value
                q = st.orientation.value
                ab = st.accel_bias.value
                wb = st.angular_bias.value

                P = st.covariance

                tf_q = quaternion_wxyz2xyzw(q)
                R_inert_body = tft.quaternion_matrix(tf_q)[0:3,0:3]

                Fx = np.eye(15)
                Fx[0:3, 3:6] = dt * np.eye(3)
                Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab)) * dt
                Fx[3:6, 9:12] = -dt * R_inert_body
                Fx[6:9, 12:15] = -dt * R_inert_body

                Qi = np.eye(12)
                Qi[0:3, 0:3] = var_am * dt ** 2 * Qi[0:3, 0:3]
                Qi[3:6, 3:6] = var_wm * dt ** 2 * Qi[3:6, 3:6]
                Qi[6:9, 6:9] = self._aw_var * dt * Qi[6:9, 6:9]
                Qi[9:12, 9:12] = self._ww_var * dt * Qi[9:12, 9:12]

                P = Fx.dot(P).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T)

                p = p + v * dt + 0.5 * (R_inert_body.dot(am - ab) + self._g) * dt ** 2
                v = v + (R_inert_body.dot(am - ab) + self._g) * dt
                # q = Quaternion(q[0],q[1],q[2],q[3]).quat_mult_left(Quaternion(axis_angle=dt * (wm - wb)),out='Quaternion').normalize().to_numpy()
                angle = tft.vector_norm(dt * (wm - wb))
                axis = tft.unit_vector(dt * (wm - wb))
                q = quaternion_xyzw2wxyz( tft.quaternion_multiply( quaternion_wxyz2xyzw(q), tft.quaternion_about_axis(angle, axis)) )
                ab = ab
                wb = wb

                st = KalmanStateObject()
                st.position.value = p
                st.position.time = time
                st.velocity.value = v
                st.velocity.time = time
                st.orientation.value = q
                st.orientation.time = time
                st.accel_bias.value = ab
                st.accel_bias.time = time
                st.angular_bias.value = wb
                st.angular_bias.time = time
                st.covariance = P
                st.accel_input = am
                st.accel_var = var_am
                st.angular_input = wm
                st.angular_var = var_wm
                st.state_time = time
                st.initialized = True
                if loadindex == -1:
                    self._state_buffer.add_state()
                    self._state_buffer.update_state(st, -1)
                else:
                    self._state_buffer.update_state(st, loadindex+1)


    def correct(self, meas_func, hx_func, V, time, measurementname='unspecified'):
        with self._lock:
            oldest = self._state_buffer.get_state(0).state_time
            latest = self._state_buffer.get_state(-1).state_time
            if time < oldest:
                log.log(measurementname, 'measurement is too old. measurement time:', time, 'filter time range:', oldest, ' to ', latest)
                return

            index = self._state_buffer.get_index_of_closest_state_in_time(time)
            buffer_length = self._state_buffer.get_buffer_length()
            if index < 0 or index >= buffer_length:
                log.log('index out of range. index:', index, 'state buf length:', buffer_length)
                return

            st = self._state_buffer.get_state(index)

            p = st.position.value
            v = st.velocity.value
            q = st.orientation.value
            ab = st.accel_bias.value
            wb = st.angular_bias.value

            P = st.covariance

            unsmooth_cov = deepcopy(P)

            Q_dtheta = 0.5 * np.array([
                [-q[1], -q[2], -q[3]],
                [ q[0],  q[3], -q[2]],
                [-q[3],  q[0],  q[1]],
                [ q[2], -q[1],  q[0]]
            ])

            X_dtheta = np.zeros([16,15])
            X_dtheta[0:6,0:6] = np.eye(6)
            X_dtheta[6:10,6:9] = Q_dtheta
            X_dtheta[10:16,9:15] = np.eye(6)

            state_as_numpy = st.values_to_numpy()[0:-2]
            Hx = hx_func(state_as_numpy)
            H = np.matmul(Hx,X_dtheta)
            K = np.matmul(np.matmul(P, H.T), np.linalg.inv(np.matmul(np.matmul(H, P), H.T) + V))
            P = np.matmul(np.eye(15) - np.matmul(K, H), P)
            P = 0.5 * (P + P.T)
            dx = K.dot(meas_func(state_as_numpy))

            p = p + dx[0:3]
            v = v + dx[3:6]
            # q = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q,out='Quaternion').normalize().to_numpy()
            angle = tft.vector_norm(dx[6:9])
            axis = tft.unit_vector(dx[6:9])
            q = quaternion_xyzw2wxyz( tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), quaternion_wxyz2xyzw(q)) )
            ab = ab + dx[9:12]
            wb = wb + dx[12:15]

            st.position.value = p
            st.position.time = time
            st.velocity.value = v
            st.velocity.time = time
            st.orientation.value = q
            st.orientation.time = time
            st.accel_bias.value = ab
            st.accel_bias.time = time
            st.angular_bias.value = wb
            st.angular_bias.time = time
            st.covariance = P
            st.state_time = time
            st.initialized = True
            self._state_buffer.update_state(st, index)

            # self.backward_smooth(index, dx, unsmooth_cov)

            if index+1 < buffer_length:
                for i in range(index+1, buffer_length):
                    ist = self._state_buffer.get_state(i)
                    self.predict(ist.accel_input, ist.accel_var, ist.angular_input, ist.angular_var, ist.state_time,
                                 i - 1, measurementname + '_correction @ ' + str(time))


    def correct_relative(self, meas_func, hx_func, V, time0, time1, measurementname='unspecified'):
        with self._lock:
            oldest = self._state_buffer.get_state(0).state_time
            latest = self._state_buffer.get_state(-1).state_time
            if time0 < oldest:
                log.log(measurementname, 'measurement is too old. measurement time:', time0, 'filter time range:', oldest, ' to ', latest)
                return

            index0 = self._state_buffer.get_index_of_closest_state_in_time(time0)
            index1 = self._state_buffer.get_index_of_closest_state_in_time(time1)
            buffer_length = self._state_buffer.get_buffer_length()

            st0 = self._state_buffer.get_state(index0)
            st1 = self._state_buffer.get_state(index1)

            p0 = st0.position.value
            v0 = st0.velocity.value
            q0 = st0.orientation.value
            ab0 = st0.accel_bias.value
            wb0 = st0.angular_bias.value
            P0 = st0.covariance

            p1 = st1.position.value
            v1 = st1.velocity.value
            q1 = st1.orientation.value
            ab1 = st1.accel_bias.value
            wb1 = st1.angular_bias.value
            P1 = st1.covariance

            Fx_prod = np.eye(15)
            for i in range(index0+1, index1+1):
                st = self._state_buffer.get_state(i)

                p = st.position.value
                v = st.velocity.value
                q = st.orientation.value
                ab = st.accel_bias.value
                wb = st.angular_bias.value
                am = st.accel_input
                dt = st.state_time - self._state_buffer.get_state(i-1).state_time

                tf_q = np.concatenate([q[1:4], [q[0]]])
                R_inert_body = tft.quaternion_matrix(tf_q)[0:3, 0:3]

                Fx = np.eye(15)
                Fx[0:3, 3:6] = dt * np.eye(3)
                Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab)) * dt
                Fx[3:6, 9:12] = -dt * R_inert_body
                Fx[6:9, 12:15] = -dt * R_inert_body

                Fx_prod = np.matmul(Fx_prod, Fx)

            P = np.zeros((30, 30))
            P[0:15, 0:15] = P0
            P[15:30, 15:30] = P1
            P[0:15, 15:30] = np.matmul(P0, Fx_prod.T)
            P[15:30, 0:15] = np.matmul(Fx_prod, P0)

            Q_dtheta0 = 0.5 * np.array([
                [-q0[1], -q0[2], -q0[3]],
                [ q0[0],  q0[3], -q0[2]],
                [-q0[3],  q0[0],  q0[1]],
                [ q0[2], -q0[1],  q0[0]]
            ])

            X_dtheta0 = np.zeros([16,15])
            X_dtheta0[0:6,0:6] = np.eye(6)
            X_dtheta0[6:10,6:9] = Q_dtheta0
            X_dtheta0[10:16,9:15] = np.eye(6)

            Q_dtheta1 = 0.5 * np.array([
                [-q1[1], -q1[2], -q1[3]],
                [ q1[0],  q1[3], -q1[2]],
                [-q1[3],  q1[0],  q1[1]],
                [ q1[2], -q1[1],  q1[0]]
            ])

            X_dtheta1 = np.zeros([16, 15])
            X_dtheta1[0:6, 0:6] = np.eye(6)
            X_dtheta1[6:10, 6:9] = Q_dtheta1
            X_dtheta1[10:16, 9:15] = np.eye(6)

            X_dtheta = np.zeros((32, 30))
            X_dtheta[0:16, 0:15] = X_dtheta0
            X_dtheta[16:32, 15:30] = X_dtheta1

            state_as_numpy = np.concatenate([st0.values_to_numpy()[0:-2], st1.values_to_numpy()[0:-2]])
            Hx = hx_func(state_as_numpy)
            H = np.matmul(Hx,X_dtheta)
            S = np.matmul(np.matmul(H, P), H.T) + V
            K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))
            K1 = K[15:30, :]
            P1 = P1 - np.matmul(np.matmul(K1, S), K1.T)
            dx = K1.dot(meas_func(state_as_numpy))
            # log.log('indices(', index0, index1, ')  dx', dx)

            p1 = p1 + dx[0:3]
            v1 = v1 + dx[3:6]
            # q1 = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q1,out='Quaternion').normalize().to_numpy()
            angle = tft.vector_norm(dx[6:9])
            axis = tft.unit_vector(dx[6:9])
            q1 = quaternion_xyzw2wxyz( tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), quaternion_wxyz2xyzw(q1)) )
            ab1 = ab1 + dx[9:12]
            wb1 = wb1 + dx[12:15]

            st1.position.value = p1
            st1.position.time = time1
            st1.velocity.value = v1
            st1.velocity.time = time1
            st1.orientation.value = q1
            st1.orientation.time = time1
            st1.accel_bias.value = ab1
            st1.accel_bias.time = time1
            st1.angular_bias.value = wb1
            st1.angular_bias.time = time1
            st1.covariance = P1
            st1.state_time = time1
            st1.initialized = True
            self._state_buffer.update_state(st1, index1)

            st0 = self._state_buffer.get_state(index0)
            st1 = self._state_buffer.get_state(index1)
            state_as_numpy = np.concatenate([st0.values_to_numpy()[0:-2], st1.values_to_numpy()[0:-2]])
            # log.log('corrected state', np.matmul(Hx, state_as_numpy))
            log.log('--------------------------------------------\n')

            if index1+1 < buffer_length:
                for i in range(index1+1, buffer_length):
                    ist = self._state_buffer.get_state(i)
                    self.predict(ist.accel_input, ist.accel_var, ist.angular_input, ist.angular_var, ist.state_time,
                                 i - 1, measurementname + '_correction @ ' + str(time1))

    def backward_smooth(self, index, dx, unsmooth_cov):
        for i in range(index, 0, -1):
            st_prev = self._state_buffer.get_state(i-1)
            st_curr = self._state_buffer.get_state(i)

            p_prev = st_prev.position.value
            v_prev = st_prev.velocity.value
            q_prev = st_prev.orientation.value
            ab_prev = st_prev.accel_bias.value
            wb_prev = st_prev.angular_bias.value

            am = st_prev.accel_input
            wm = st_prev.angular_input

            P_prev = st_prev.covariance
            P_curr = st_curr.covariance

            dt = st_curr.state_time - st_prev.state_time

            tf_q = np.concatenate([q_prev[1:4], [q_prev[0]]])
            R_inert_body = tft.quaternion_matrix(tf_q)[0:3, 0:3]

            Fx = np.eye(15)
            Fx[0:3, 3:6] = dt * np.eye(3)
            Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab_prev)) * dt
            Fx[3:6, 9:12] = -dt * R_inert_body
            Fx[6:9, 12:15] = -dt * R_inert_body

            if np.linalg.det(unsmooth_cov) == 0:
                log.log('non-invertible covariance matrix')
                log.log(unsmooth_cov)
                log.log('\n\n')

            A = np.matmul(P_prev, np.matmul(Fx.T, np.linalg.inv(unsmooth_cov)))
            dx = np.matmul(A, dx)
            new_unsmooth_cov = deepcopy(P_prev)
            P_prev = P_prev + np.matmul(A, np.matmul(P_curr - unsmooth_cov, A.T))
            P_prev = 0.5 * (P_prev + P_prev.T)
            unsmooth_cov = new_unsmooth_cov

            p_prev = p_prev + dx[0:3]
            v_prev = v_prev + dx[3:6]
            # q_prev = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q_prev, out='Quaternion').normalize().to_numpy()
            angle = tft.vector_norm(dx[6:9])
            axis = tft.unit_vector(dx[6:9])
            q_prev = quaternion_xyzw2wxyz(
                tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), quaternion_wxyz2xyzw(q_prev)))
            ab_prev = ab_prev + dx[9:12]
            wb_prev = wb_prev + dx[12:15]

            st_prev.position.value = p_prev
            st_prev.velocity.value = v_prev
            st_prev.orientation.value = q_prev
            st_prev.accel_bias.value = ab_prev
            st_prev.angular_bias.value = wb_prev
            st_prev.covariance = P_prev
            self._state_buffer.update_state(st_prev, i-1)


    def get_state_as_numpy(self):
        with self._lock:
            return self._state_buffer.get_state(-1).values_to_numpy()


    def is_initialized(self):
        with self._lock:
            return self._state_buffer.get_state(-1).initialized


    def get_covariance_as_numpy(self):
        with self._lock:
            P = self._state_buffer.get_state(-1).covariance
            return P.flatten()
