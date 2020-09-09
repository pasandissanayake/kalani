''' version 1.0.0 - modified to be compatible with python 2 '''
import tf.transformations as tft

import numpy as np
import threading
from rotations_v1 import skew_symmetric, Quaternion
from state_buffer_v1 import StateBuffer, StateObject


class Kalman_Filter_V1():
    def __init__(self, g, aw_var, ww_var):
        # State buffer length
        self.STATE_BUFFER_LENGTH = 50

        # Allowed maximum gap between any two initial state variables (in seconds)
        self.STATE_INIT_TIME_THRESHOLD = 1

        self._state_buffer = StateBuffer(self.STATE_BUFFER_LENGTH)
        self._lock = threading.RLock()

        self._aw_var = aw_var
        self._ww_var = ww_var
        self._g = g

        # Prediction matrices
        self.Fi = np.zeros([15, 12])
        self.Fi[3:15, :] = np.eye(12)


    def initialize_state(self, p=None, cov_p=None, v=None, cov_v=None, q=None, cov_q=None, ab=None, cov_ab=None, wb=None, cov_wb=None, g=None, time=-1):
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
                    print('position covariances are not provided')

            if v is not None:
                if cov_v is not None:
                    st.velocity.value = np.array(v)
                    st.velocity.time = time
                    st.covariance[3:6, 3:6] = np.diag(cov_v)
                    self._state_buffer.update_state(st, -1)
                else:
                    print('velocity covariances are not provided')

            if q is not None:
                if cov_q is not None:
                    st.orientation.value = np.array(q)
                    st.orientation.time = time
                    st.covariance[6:9, 6:9] = np.diag(cov_q)
                    self._state_buffer.update_state(st, -1)
                else:
                    print('rotation covariances are not provided')

            if ab is not None:
                if cov_ab is not None:
                    st.accel_bias.value = np.array(ab)
                    st.accel_bias.time = time
                    st.covariance[9:12, 9:12] = np.diag(cov_ab)
                    self._state_buffer.update_state(st, -1)
                else:
                    print('acceleration bias covariances are not provided')

            if wb is not None:
                if cov_wb is not None:
                    st.angular_bias.value = np.array(wb)
                    st.angular_bias.time = time
                    st.covariance[0:3, 0:3] = np.diag(cov_wb)
                    self._state_buffer.update_state(st, -1)
                else:
                    print('angular velocity bias covariances are not provided')

            st = self._state_buffer.get_state(-1)
            timestamps = st.times_to_numpy()
            initialized = all(ts >= 0 for ts in timestamps)
            if self._state_buffer.get_timestamp_of_newest_variable(-1) - self._state_buffer.get_timestamp_of_oldest_variable(-1) < self.STATE_INIT_TIME_THRESHOLD and initialized:
                st.state_time = time
                st.initialized = True
                self._state_buffer.update_state(st, -1)
                print('state initialized')
            else:
                print('initial state time stamps:', timestamps)


    def predict(self, am, var_am, wm, var_wm, time, loadindex=-1, inputname='unspecified'):
        with self._lock:
            st = self._state_buffer.get_state(loadindex)
            if st is None or not st.initialized:
                print('state not initialized')
                return

            st_time = st.state_time
            if st_time >= time:
                print(inputname, 'input is too old. input time:', time, 'filter time:', st_time, 'load index:', loadindex)
                return

            else:
                dt = time - st_time

                p = st.position.value
                v = st.velocity.value
                q = st.orientation.value
                ab = st.accel_bias.value
                wb = st.angular_bias.value

                P = st.covariance

                # R_inert_body = Quaternion(q[0],q[1],q[2],q[3]).to_mat()
                tf_q = np.concatenate([q[1:4],[q[0]]])
                R_inert_body = tft.quaternion_matrix(tf_q)[0:3,0:3]

                # print 'quater(wxyz):', q
                # print 'axis angle:', Quaternion(q[0],q[1],q[2],q[3]).to_axis_angle()
                # print 'euler (rpy):', Quaternion(q[0],q[1],q[2],q[3]).to_euler()
                # print 'tf euler (rpy):', tft.euler_from_quaternion(tf_q,axes='sxyz')

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
                q = Quaternion(q[0],q[1],q[2],q[3]).quat_mult_left(Quaternion(axis_angle=dt * (wm - wb)),out='Quaternion').normalize().to_numpy()
                ab = ab
                wb = wb

                # print 'am:', am
                # print 'converted am:', R_inert_body.dot(am)
                # print 'converted am+g:', R_inert_body.dot(am) + self._g
                # print 'R:', R_inert_body
                # print 'converted ve:', v
                # print 'ab:', ab
                # print '------------------------------------------------------------\n'

                st = StateObject()
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
                print(measurementname, 'measurement is too old. measurement time:', time, 'filter time range:', oldest, ' to ', latest)
                return

            index = self._state_buffer.get_index_of_closest_state_in_time(time)
            buffer_length = self._state_buffer.get_buffer_length()
            if index < 0 or index >= buffer_length:
                print('index out of range. index:', index, 'state buf length:', buffer_length)
                return

            st = self._state_buffer.get_state(index)

            p = st.position.value
            v = st.velocity.value
            q = st.orientation.value
            ab = st.accel_bias.value
            wb = st.angular_bias.value

            P = st.covariance

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
            dx = K.dot(meas_func(state_as_numpy))
            # dx = K.dot(y - np.matmul(Hx, state_as_numpy))

            p = p + dx[0:3]
            v = v + dx[3:6]
            q = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q,out='Quaternion').normalize().to_numpy()
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
                print(measurementname, 'measurement is too old. measurement time:', time0, 'filter time range:', oldest, ' to ', latest)
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
            # print 'indices(', index0, index1, ')  dx', dx

            p1 = p1 + dx[0:3]
            v1 = v1 + dx[3:6]
            q1 = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q1,out='Quaternion').normalize().to_numpy()
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
            print 'corrected state', np.matmul(Hx, state_as_numpy)
            print '--------------------------------------------\n'

            if index1+1 < buffer_length:
                for i in range(index1+1, buffer_length):
                    ist = self._state_buffer.get_state(i)
                    self.predict(ist.accel_input, ist.accel_var, ist.angular_input, ist.angular_var, ist.state_time,
                                 i - 1, measurementname + '_correction @ ' + str(time1))

    def backward_smooth(self, index):
        for i in range(index, 0, -1):
            pass



    # def correct_relative(self, meas_func, hx_func, V, time0, time1, measurementname='unspecified'):
    #     with self._lock:
    #         oldest = self._state_buffer.get_state(0).state_time
    #         latest = self._state_buffer.get_state(-1).state_time
    #         if time0 < oldest:
    #             print(measurementname, 'measurement is too old. measurement time:', time0, 'filter time range:', oldest, ' to ', latest)
    #             return
    #
    #         index0 = self._state_buffer.get_index_of_closest_state_in_time(time0)
    #         index1 = self._state_buffer.get_index_of_closest_state_in_time(time1)
    #         buffer_length = self._state_buffer.get_buffer_length()
    #
    #         st0 = self._state_buffer.get_state(index0)
    #         st1 = self._state_buffer.get_state(index1)
    #
    #         p0 = st0.position.value
    #         v0 = st0.velocity.value
    #         q0 = st0.orientation.value
    #         ab0 = st0.accel_bias.value
    #         wb0 = st0.angular_bias.value
    #         P0 = st0.covariance
    #
    #         p1 = st1.position.value
    #         v1 = st1.velocity.value
    #         q1 = st1.orientation.value
    #         ab1 = st1.accel_bias.value
    #         wb1 = st1.angular_bias.value
    #         P1 = st1.covariance
    #
    #         Fx_prod = np.eye(15)
    #         for i in range(index0+1, index1+1):
    #             st = self._state_buffer.get_state(i)
    #
    #             p = st.position.value
    #             v = st.velocity.value
    #             q = st.orientation.value
    #             ab = st.accel_bias.value
    #             wb = st.angular_bias.value
    #             am = st.accel_input
    #             dt = st.state_time - self._state_buffer.get_state(i-1).state_time
    #
    #             tf_q = np.concatenate([q[1:4], [q[0]]])
    #             R_inert_body = tft.quaternion_matrix(tf_q)[0:3, 0:3]
    #
    #             Fx = np.eye(15)
    #             Fx[0:3, 3:6] = dt * np.eye(3)
    #             Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab)) * dt
    #             Fx[3:6, 9:12] = -dt * R_inert_body
    #             Fx[6:9, 12:15] = -dt * R_inert_body
    #
    #             Fx_prod = np.matmul(Fx_prod, Fx)
    #
    #         Q_dtheta0 = 0.5 * np.array([
    #             [-q0[1], -q0[2], -q0[3]],
    #             [q0[0], q0[3], -q0[2]],
    #             [-q0[3], q0[0], q0[1]],
    #             [q0[2], -q0[1], q0[0]]
    #         ])
    #
    #         X_dtheta0 = np.zeros([16, 15])
    #         X_dtheta0[0:6, 0:6] = np.eye(6)
    #         X_dtheta0[6:10, 6:9] = Q_dtheta0
    #         X_dtheta0[10:16, 9:15] = np.eye(6)
    #
    #         qw, qx, qy, qz = q0
    #         G= np.array([
    #             [qw**2 + qx**2 - qy**2 - qz**2,             2*qx*qy - 2*qw*qz,             2*qx*qz + 2*qw*qy, 0,0,0,0,0,0,0,0,0,0,0,0,0],
    #             [            2*qx*qy + 2*qw*qz, qw**2 - qx**2 + qy**2 - qz**2,             2*qy*qz - 2*qw*qx, 0,0,0,0,0,0,0,0,0,0,0,0,0],
    #             [            2*qx*qz - 2*qw*qy,             2*qy*qz + 2*qw*qx, qw**2 - qx**2 - qy**2 + qz**2, 0,0,0,0,0,0,0,0,0,0,0,0,0]
    #         ])
    #
    #         G = np.matmul(G, X_dtheta0)
    #
    #         S = P1 - np.matmul(np.matmul(Fx_prod, P0), Fx_prod.T) + np.matmul(np.matmul(G.T, V), G)
    #         K = np.matmul(P1 - np.matmul(np.matmul(Fx_prod, P0), Fx_prod.T), np.linalg.inv(S))
    #
    #         state_as_numpy = np.concatenate([st0.values_to_numpy()[0:-2], st1.values_to_numpy()[0:-2]])
    #         dx = np.matmul(K, np.matmul(G, meas_func(state_as_numpy)))
    #         # print 'indices(', index0, index1, ')  dx', dx
    #
    #         p1 = p1 + dx[0:3]
    #         v1 = v1 + dx[3:6]
    #         q1 = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q1,out='Quaternion').normalize().to_numpy()
    #         ab1 = ab1 + dx[9:12]
    #         wb1 = wb1 + dx[12:15]
    #
    #         st1.position.value = p1
    #         st1.position.time = time1
    #         st1.velocity.value = v1
    #         st1.velocity.time = time1
    #         st1.orientation.value = q1
    #         st1.orientation.time = time1
    #         st1.accel_bias.value = ab1
    #         st1.accel_bias.time = time1
    #         st1.angular_bias.value = wb1
    #         st1.angular_bias.time = time1
    #         st1.covariance = P1
    #         st1.state_time = time1
    #         st1.initialized = True
    #         self._state_buffer.update_state(st1, index1)
    #
    #         st0 = self._state_buffer.get_state(index0)
    #         st1 = self._state_buffer.get_state(index1)
    #         state_as_numpy = np.concatenate([st0.values_to_numpy()[0:-2], st1.values_to_numpy()[0:-2]])
    #         print 'corrected state', np.matmul(Hx, state_as_numpy)
    #         print '--------------------------------------------\n'
    #
    #         if index1+1 < buffer_length:
    #             for i in range(index1+1, buffer_length):
    #                 ist = self._state_buffer.get_state(i)
    #                 self.predict(ist.accel_input, ist.accel_var, ist.angular_input, ist.angular_var, ist.state_time,
    #                              i - 1, measurementname + '_correction @ ' + str(time1))


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
