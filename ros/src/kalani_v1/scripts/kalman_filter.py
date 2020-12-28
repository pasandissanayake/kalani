import numpy as np
import sympy as sp
import numdifftools as nd
import threading
from copy import deepcopy
from types import MethodType

import tf.transformations as tft

from utilities import *


log = Log('kalman_filter')
kf_config = get_config_dict()['kalman_filter']


class sarray(np.ndarray):
    def __new__(cls, template, val=None):
        length = sum([s[1] for s in template])
        obj = np.zeros(length).view(cls)
        obj.template = template
        return obj

    def __init__(self, template=None, val=None):
        i = 0
        for t in self.template:
            self._set_sub_state(i, t[0], t[1])
            i += t[1]
        if val is not None:
            self[:] = np.array(val)

    def _set_sub_state(self, idx, name, size):
        def fun():
            return self[idx:idx + size]
        setattr(self, name, fun)


class parray(np.ndarray):
    def __new__(cls, template, val=None):
        length = sum([s[1] for s in template])
        obj = np.eye(length).view(cls)
        obj.template = template
        return obj

    def __init__(self, template=None, val=None):
        i = 0
        for t in self.template:
            self._set_sub_state(i, t[0], t[1])
            i += t[1]
        if val is not None:
            self[:] = np.array(val)

    def _set_sub_state(self, idx, name, size):
        def fun():
            return self[idx:idx + size, idx:idx + size]
        setattr(self, name, fun)


class StateObject:
    def __init__(self, ns_template, es_template, mmi_template):
        self.ns = sarray(ns_template)
        self.es = sarray(es_template)
        self.es_cov = parray(es_template)
        self.predicted_ns = sarray(ns_template)
        self.predicted_es = sarray(es_template)
        self.predicted_es_cov = parray(es_template)
        self.mm_inputs = sarray(mmi_template)
        self.mm_inputs_cov = sarray(mmi_template)
        self.Fx = None
        self.Fi = None
        self.timestamp = None
        self.is_valid = False


class StateBuffer:
    def __init__(self, bufferlength):
        self._lock = threading.Lock()

        # no. of previous states stored
        self._BUFFER_LENGTH = bufferlength

        # buffer for storing state variables. lowest index for oldest state
        self._buffer = []


    def add_state(self, stateobject):
        with self._lock:
            self._buffer.append(stateobject)
            if len(self._buffer) > self._BUFFER_LENGTH:
                self._buffer.pop(0)

    def update_state(self, stateobject, index=-1):
        with self._lock:
            bl = len(self._buffer)
            if index >= bl or index < -bl:
                log.log('Store index: ' + str(index) + ' is out of range.')
            else:
                self._buffer[index] = stateobject

    def get_state(self, index):
        with self._lock:
            bl = len(self._buffer)
            if index >= bl or index < -bl:
                log.log('Store index: ' + str(index) + ' is out of range.')
                return None
            else:
                return self._buffer[index]

    def get_index_of_closest_state_in_time(self, timestamp):
        with self._lock:
            state_timestamps = [st.timestamp for st in self._buffer]
            index = min(range(len(state_timestamps)), key=lambda i: abs(state_timestamps[i] - timestamp))
            return index

    def get_buffer_length(self):
        with self._lock:
            return len(self._buffer)


class KalmanFilter:
    def __init__(self, ns_template, es_template, pn_template, mmi_template, motion_model, combination, difference):
        # State buffer length
        self.STATE_BUFFER_LENGTH = kf_config['buffer_size']

        # Allowed maximum gap between any two initial state variables (in seconds)
        self.STATE_INIT_TIME_THRESHOLD = kf_config['state_init_time_threshold']

        self._g = np.array([0, 0, kf_config['gravity']])

        self._state_buffer = StateBuffer(self.STATE_BUFFER_LENGTH)
        self._lock = threading.RLock()

        self._ns_template = ns_template
        self._es_template = es_template
        self._pn_template = pn_template
        self._mmi_template = mmi_template

        self._motion_model = motion_model
        self._combination = combination
        self._difference = difference

        self._ns_timestamps = {s[0]:-1 for s in ns_template}

        def true_state(es, ns):
            ns = sarray(ns_template, ns)
            es = sarray(es_template, es)
            return combination(ns, es)

        def fx(es, ns, pn, mmi, dt):
            pn = sarray(pn_template, pn)
            mmi = sarray(mmi_template, mmi)
            ts = sarray(ns_template, true_state(es, ns))
            return motion_model(ts, mmi, pn, dt)

        def fi(pn, es, ns, mmi, dt):
            return fx(es, ns, pn, mmi, dt)

        self._truestate_jacob = nd.Jacobian(true_state)
        sw = Stopwatch()
        sw.start()
        self._Fx_jacob = nd.Jacobian(fx)
        self._Fi_jacob = nd.Jacobian(fi)
        print 'calculation time:', sw.stop(), 's'


    def initialize(self, vals):
        # check whether a state object is available within the buffer
        if self._state_buffer.get_buffer_length() > 0:
            so = self._state_buffer.get_state(0)
            update = True
        else:
            so = StateObject(self._ns_template, self._es_template, self._mmi_template)
            update = False
        key_min = min(self._ns_timestamps.keys(), key=(lambda k: self._ns_timestamps[k]))
        val_min = self._ns_timestamps[key_min]
        for val in vals:
            # reject if timestamp is older than the current oldest timestamp
            if val[2] < val_min:
                continue
            fun = getattr(so.predicted_ns, val[0])
            fun()[:] = np.array(val[1])
            self._ns_timestamps[val[0]] = val[2]
        # check whether all the timestamps are within the threshold value from the latest timestamp
        is_initialized = True
        key_max = max(self._ns_timestamps.keys(), key=(lambda k: self._ns_timestamps[k]))
        val_max = self._ns_timestamps[key_max]
        for k in self._ns_timestamps.keys():
            if self._ns_timestamps[k] < val_max - self.STATE_INIT_TIME_THRESHOLD:
                self._ns_timestamps[k] = -1
                is_initialized = False
        so.timestamp = val_max
        so.is_valid = is_initialized
        # store the modified state object in buffer
        if update:
            self._state_buffer.update_state(so)
        else:
            self._state_buffer.add_state(so)

        pso = self._state_buffer.get_state(0)
        print pso.predicted_ns, pso.is_valid, pso.timestamp, self._ns_timestamps
        if pso.is_valid:
            sw = Stopwatch()
            sw.start()
            self._Fx_jacob(np.zeros(15), pso.predicted_ns, np.zeros(12), np.ones(6), pso.timestamp)
            self._Fi_jacob(np.zeros(12), np.zeros(15), pso.predicted_ns, np.ones(6), pso.timestamp)
            print 'substitution time:', sw.stop(), 's'





# motion model
def motion_model(ts, mmi, pn, dt):
    gravity = np.array([0, 0, -9.8])
    R = tft.quaternion_matrix(ts.q())[0:3,0:3]
    acc = (np.matmul(R, (mmi.a()-ts.ab())) + gravity)
    dtheta = (mmi.w() - ts.wb()) * dt + pn.q()
    axis = tft.unit_vector(dtheta)
    angle = tft.vector_norm(dtheta)
    res = [
        ts.p() + ts.v() * dt + 0.5 * acc * dt**2,
        ts.v() + acc * dt + pn.v(),
        tft.quaternion_multiply(ts.q(), tft.quaternion_about_axis(angle, axis)),
        ts.ab() + pn.ab(),
        ts.wb() + pn.wb()
    ]
    return np.concatenate(res)


# nominal state, error state conjugation
# nominal state + error state ==> nominal state
def combination(ns, es):
    eulers = es.q()
    res = [
        ns.p() + es.p(),
        ns.v() + es.v(),
        tft.quaternion_multiply(ns.q(), tft.quaternion_from_euler(eulers[0], eulers[1], eulers[2])),
        ns.ab() + es.ab(),
        ns.wb() + es.wb()
    ]
    return np.concatenate(res)


# nominal state difference
# nominal state(ns1) - nominal state(ns0) ==> error state
def difference(ns1, ns0):
    res = [
        ns1.p() - ns0.p(),
        ns1.v() - ns0.v(),
        tft.euler_from_quaternion(tft.quaternion_multiply(tft.quaternion_conjugate(ns0.q()), ns1.q())),
        ns1.ab() - ns0.ab(),
        ns1.wb() - ns0.wb()
    ]
    return np.concatenate(res)


kf = KalmanFilter(
    # nominal/true state template
    [
        ['p', 3],
        ['v', 3],
        ['q', 4],
        ['ab', 3],
        ['wb', 3]
    ],

    # error state template
    [
        ['p', 3],
        ['v', 3],
        ['q', 3],
        ['ab', 3],
        ['wb', 3]
    ],

    # process noise template
    [
        ['v', 3],
        ['q', 3],
        ['ab', 3],
        ['wb', 3]
    ],

    # motion model input template
    [
        ['a', 3],
        ['w', 3]
    ],

    motion_model,
    combination,
    difference
)


kf.initialize([
    ['p', [1, 2, 3], 1.7]
])

kf.initialize([
    ['q', [1, 0, 0, 0], 1.7]
])

kf.initialize([
    ['v', [3,3,3,], 2.0],
    ['ab', [4,4,4], 2.0],
    ['wb', [2,2,2], 2.0]
])


# class KalmanFilter():
#     def __init__(self):
#         config = get_config_dict()['kalman_filter']
#
#         # State buffer length
#         self.STATE_BUFFER_LENGTH = config['buffer_size']
#
#         # Allowed maximum gap between any two initial state variables (in seconds)
#         self.STATE_INIT_TIME_THRESHOLD = config['state_init_time_threshold']
#
#         self._g = np.array([0, 0, config['gravity']])
#
#         self._state_buffer = KalmanBuffer(self.STATE_BUFFER_LENGTH)
#         self._lock = threading.RLock()
#
#         # Prediction matrices
#         self.Fi = np.zeros([15, 12])
#         self.Fi[3:15, :] = np.eye(12)
#
#
#     def initialize_state(self, p=None, cov_p=None, v=None, cov_v=None, q=None, cov_q=None, ab=None, cov_ab=None, wb=None, cov_wb=None, time=-1):
#         with self._lock:
#             if self._state_buffer.get_buffer_length() < 1: self._state_buffer.add_state()
#
#             st = self._state_buffer.get_state(-1)
#             if p is not None:
#                 if cov_p is not None:
#                     st.position.value = np.array(p)
#                     st.position.time = time
#                     st.covariance[0:3,0:3] = np.diag(cov_p)
#                     self._state_buffer.update_state(st,-1)
#                 else:
#                     log.log('position covariances are not provided')
#
#             if v is not None:
#                 if cov_v is not None:
#                     st.velocity.value = np.array(v)
#                     st.velocity.time = time
#                     st.covariance[3:6, 3:6] = np.diag(cov_v)
#                     self._state_buffer.update_state(st, -1)
#                 else:
#                     log.log('velocity covariances are not provided')
#
#             if q is not None:
#                 if cov_q is not None:
#                     st.orientation.value = np.array(q)
#                     st.orientation.time = time
#                     st.covariance[6:9, 6:9] = np.diag(cov_q)
#                     self._state_buffer.update_state(st, -1)
#                 else:
#                     log.log('rotation covariances are not provided')
#
#             if ab is not None:
#                 if cov_ab is not None:
#                     st.accel_bias.value = np.array(ab)
#                     st.accel_bias.time = time
#                     st.covariance[9:12, 9:12] = np.diag(cov_ab)
#                     self._state_buffer.update_state(st, -1)
#                 else:
#                     log.log('acceleration bias covariances are not provided')
#
#             if wb is not None:
#                 if cov_wb is not None:
#                     st.angular_bias.value = np.array(wb)
#                     st.angular_bias.time = time
#                     st.covariance[12:15, 12:15] = np.diag(cov_wb)
#                     self._state_buffer.update_state(st, -1)
#                 else:
#                     log.log('angular velocity bias covariances are not provided')
#
#             st = self._state_buffer.get_state(-1)
#             timestamps = st.times_to_numpy()
#             initialized = all(ts >= 0 for ts in timestamps)
#             if self._state_buffer.get_timestamp_of_newest_variable(-1) - self._state_buffer.get_timestamp_of_oldest_variable(-1) < self.STATE_INIT_TIME_THRESHOLD and initialized:
#                 st.state_time = time
#                 st.initialized = True
#                 self._state_buffer.update_state(st, -1)
#                 log.log('state initialized')
#             else:
#                 log.log('initial state time stamps:', timestamps)
#
#
#     def predict(self, am, var_am, wm, var_wm, var_aw, var_ww, time, loadindex=-1, inputname='unspecified'):
#         with self._lock:
#             st = self._state_buffer.get_state(loadindex)
#             if st is None or not st.initialized:
#                 log.log('state not initialized')
#                 return
#
#             st_time = st.state_time
#             if st_time >= time:
#                 log.log(inputname, 'input is too old. input time:', time, 'filter time:', st_time, 'load index:', loadindex)
#                 return
#
#             else:
#                 dt = time - st_time
#
#                 p = st.position.value
#                 v = st.velocity.value
#                 q = st.orientation.value
#                 ab = st.accel_bias.value
#                 wb = st.angular_bias.value
#
#                 P = st.covariance
#
#                 tf_q = quaternion_wxyz2xyzw(q)
#                 R_inert_body = tft.quaternion_matrix(tf_q)[0:3,0:3]
#
#                 Fx = np.eye(15)
#                 Fx[0:3, 3:6] = dt * np.eye(3)
#                 Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab)) * dt
#                 Fx[3:6, 9:12] = -dt * R_inert_body
#                 Fx[6:9, 12:15] = -dt * R_inert_body
#
#                 Qi = np.eye(12)
#                 Qi[0:3, 0:3] = var_am * dt ** 2
#                 Qi[3:6, 3:6] = var_wm * dt ** 2
#                 Qi[6:9, 6:9] = var_aw * dt
#                 Qi[9:12, 9:12] = var_ww * dt
#
#                 P = Fx.dot(P).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T)
#
#                 p = p + v * dt + 0.5 * (R_inert_body.dot(am - ab) + self._g) * dt ** 2
#                 v = v + (R_inert_body.dot(am - ab) + self._g) * dt
#                 q = Quaternion(q[0],q[1],q[2],q[3]).quat_mult_left(Quaternion(axis_angle=dt * (wm - wb)),out='Quaternion').normalize().to_numpy()
#                 angle = tft.vector_norm(dt * (wm - wb))
#                 axis = tft.unit_vector(dt * (wm - wb))
#                 # q = quaternion_xyzw2wxyz( tft.quaternion_multiply( quaternion_wxyz2xyzw(q), tft.quaternion_about_axis(angle, axis)) )
#                 ab = ab
#                 wb = wb
#
#                 st = KalmanStateObject()
#                 st.position.value = p
#                 st.position.time = time
#                 st.velocity.value = v
#                 st.velocity.time = time
#                 st.orientation.value = q
#                 st.orientation.time = time
#                 st.accel_bias.value = ab
#                 st.accel_bias.time = time
#                 st.angular_bias.value = wb
#                 st.angular_bias.time = time
#                 st.covariance = P
#                 st.accel_input = am
#                 st.accel_var = var_am
#                 st.angular_input = wm
#                 st.angular_var = var_wm
#                 st.state_time = time
#                 st.initialized = True
#                 if loadindex == -1:
#                     self._state_buffer.add_state()
#                     self._state_buffer.update_state(st, -1)
#                 else:
#                     self._state_buffer.update_state(st, loadindex+1)
#
#
#     def correct(self, meas_func, hx_func, V, time, measurementname='unspecified'):
#         with self._lock:
#             oldest = self._state_buffer.get_state(0).state_time
#             latest = self._state_buffer.get_state(-1).state_time
#             if time < oldest:
#                 log.log(measurementname, 'measurement is too old. measurement time:', time, 'filter time range:', oldest, ' to ', latest)
#                 return
#
#             index = self._state_buffer.get_index_of_closest_state_in_time(time)
#             buffer_length = self._state_buffer.get_buffer_length()
#             if index < 0 or index >= buffer_length:
#                 log.log('index out of range. index:', index, 'state buf length:', buffer_length)
#                 return
#
#             st = self._state_buffer.get_state(index)
#
#             p = st.position.value
#             v = st.velocity.value
#             q = st.orientation.value
#             ab = st.accel_bias.value
#             wb = st.angular_bias.value
#
#             P = st.covariance
#
#             unsmooth_cov = deepcopy(P)
#
#             Q_dtheta = 0.5 * np.array([
#                 [-q[1], -q[2], -q[3]],
#                 [ q[0],  q[3], -q[2]],
#                 [-q[3],  q[0],  q[1]],
#                 [ q[2], -q[1],  q[0]]
#             ])
#
#             X_dtheta = np.zeros([16,15])
#             X_dtheta[0:6,0:6] = np.eye(6)
#             X_dtheta[6:10,6:9] = Q_dtheta
#             X_dtheta[10:16,9:15] = np.eye(6)
#
#             state_as_numpy = st.values_to_numpy()[0:-2]
#             Hx = hx_func(state_as_numpy)
#             H = np.matmul(Hx,X_dtheta)
#             K = np.matmul(np.matmul(P, H.T), np.linalg.inv(np.matmul(np.matmul(H, P), H.T) + V))
#             P = np.matmul(np.eye(15) - np.matmul(K, H), P)
#             P = 0.5 * (P + P.T)
#             dx = K.dot(meas_func(state_as_numpy))
#
#             p = p + dx[0:3]
#             v = v + dx[3:6]
#             # q = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q,out='Quaternion').normalize().to_numpy()
#             angle = tft.vector_norm(dx[6:9])
#             axis = np.reshape(tft.unit_vector(dx[6:9]), (3))
#             q = quaternion_xyzw2wxyz( tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), quaternion_wxyz2xyzw(q)) )
#             ab = ab + dx[9:12]
#             wb = wb + dx[12:15]
#
#             st.position.value = p
#             st.position.time = time
#             st.velocity.value = v
#             st.velocity.time = time
#             st.orientation.value = q
#             st.orientation.time = time
#             st.accel_bias.value = ab
#             st.accel_bias.time = time
#             st.angular_bias.value = wb
#             st.angular_bias.time = time
#             st.covariance = P
#             st.state_time = time
#             st.initialized = True
#             self._state_buffer.update_state(st, index)
#
#             # self.backward_smooth(index, dx, unsmooth_cov)
#
#             if index+1 < buffer_length:
#                 for i in range(index+1, buffer_length):
#                     ist = self._state_buffer.get_state(i)
#                     self.predict(ist.accel_input, ist.accel_var, ist.angular_input, ist.angular_var, ist.covariance[9:12,9:12],
#                                  ist.covariance[12:15, 12:15], ist.state_time, i - 1, measurementname + '_correction @ ' + str(time))
#
#
#     def correct_relative(self, meas_func, hx_func, V, time0, time1, measurementname='unspecified'):
#         with self._lock:
#             oldest = self._state_buffer.get_state(0).state_time
#             latest = self._state_buffer.get_state(-1).state_time
#             if time0 < oldest:
#                 log.log(measurementname, 'measurement is too old. measurement time:', time0, 'filter time range:', oldest, ' to ', latest)
#                 return
#
#             index0 = self._state_buffer.get_index_of_closest_state_in_time(time0)
#             index1 = self._state_buffer.get_index_of_closest_state_in_time(time1)
#             buffer_length = self._state_buffer.get_buffer_length()
#
#             st0 = self._state_buffer.get_state(index0)
#             st1 = self._state_buffer.get_state(index1)
#
#             p0 = st0.position.value
#             v0 = st0.velocity.value
#             q0 = st0.orientation.value
#             ab0 = st0.accel_bias.value
#             wb0 = st0.angular_bias.value
#             P0 = st0.covariance
#
#             p1 = st1.position.value
#             v1 = st1.velocity.value
#             q1 = st1.orientation.value
#             ab1 = st1.accel_bias.value
#             wb1 = st1.angular_bias.value
#             P1 = st1.covariance
#
#             Fx_prod = np.eye(15)
#             for i in range(index0+1, index1+1):
#                 st = self._state_buffer.get_state(i)
#
#                 p = st.position.value
#                 v = st.velocity.value
#                 q = st.orientation.value
#                 ab = st.accel_bias.value
#                 wb = st.angular_bias.value
#                 am = st.accel_input
#                 dt = st.state_time - self._state_buffer.get_state(i-1).state_time
#
#                 tf_q = np.concatenate([q[1:4], [q[0]]])
#                 R_inert_body = tft.quaternion_matrix(tf_q)[0:3, 0:3]
#
#                 Fx = np.eye(15)
#                 Fx[0:3, 3:6] = dt * np.eye(3)
#                 Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab)) * dt
#                 Fx[3:6, 9:12] = -dt * R_inert_body
#                 Fx[6:9, 12:15] = -dt * R_inert_body
#
#                 Fx_prod = np.matmul(Fx_prod, Fx)
#
#             P = np.zeros((30, 30))
#             P[0:15, 0:15] = P0
#             P[15:30, 15:30] = P1
#             P[0:15, 15:30] = np.matmul(P0, Fx_prod.T)
#             P[15:30, 0:15] = np.matmul(Fx_prod, P0)
#
#             Q_dtheta0 = 0.5 * np.array([
#                 [-q0[1], -q0[2], -q0[3]],
#                 [ q0[0],  q0[3], -q0[2]],
#                 [-q0[3],  q0[0],  q0[1]],
#                 [ q0[2], -q0[1],  q0[0]]
#             ])
#
#             X_dtheta0 = np.zeros([16,15])
#             X_dtheta0[0:6,0:6] = np.eye(6)
#             X_dtheta0[6:10,6:9] = Q_dtheta0
#             X_dtheta0[10:16,9:15] = np.eye(6)
#
#             Q_dtheta1 = 0.5 * np.array([
#                 [-q1[1], -q1[2], -q1[3]],
#                 [ q1[0],  q1[3], -q1[2]],
#                 [-q1[3],  q1[0],  q1[1]],
#                 [ q1[2], -q1[1],  q1[0]]
#             ])
#
#             X_dtheta1 = np.zeros([16, 15])
#             X_dtheta1[0:6, 0:6] = np.eye(6)
#             X_dtheta1[6:10, 6:9] = Q_dtheta1
#             X_dtheta1[10:16, 9:15] = np.eye(6)
#
#             X_dtheta = np.zeros((32, 30))
#             X_dtheta[0:16, 0:15] = X_dtheta0
#             X_dtheta[16:32, 15:30] = X_dtheta1
#
#             state_as_numpy = np.concatenate([st0.values_to_numpy()[0:-2], st1.values_to_numpy()[0:-2]])
#             Hx = hx_func(state_as_numpy)
#             H = np.matmul(Hx,X_dtheta)
#             S = np.matmul(np.matmul(H, P), H.T) + V
#             K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))
#             K1 = K[15:30, :]
#             P1 = P1 - np.matmul(np.matmul(K1, S), K1.T)
#             dx = K1.dot(meas_func(state_as_numpy))
#             # log.log('indices(', index0, index1, ')  dx', dx)
#
#             p1 = p1 + dx[0:3]
#             v1 = v1 + dx[3:6]
#             # q1 = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q1,out='Quaternion').normalize().to_numpy()
#             angle = tft.vector_norm(dx[6:9])
#             axis = tft.unit_vector(dx[6:9])
#             q1 = quaternion_xyzw2wxyz( tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), quaternion_wxyz2xyzw(q1)) )
#             ab1 = ab1 + dx[9:12]
#             wb1 = wb1 + dx[12:15]
#
#             st1.position.value = p1
#             st1.position.time = time1
#             st1.velocity.value = v1
#             st1.velocity.time = time1
#             st1.orientation.value = q1
#             st1.orientation.time = time1
#             st1.accel_bias.value = ab1
#             st1.accel_bias.time = time1
#             st1.angular_bias.value = wb1
#             st1.angular_bias.time = time1
#             st1.covariance = P1
#             st1.state_time = time1
#             st1.initialized = True
#             self._state_buffer.update_state(st1, index1)
#
#             st0 = self._state_buffer.get_state(index0)
#             st1 = self._state_buffer.get_state(index1)
#             state_as_numpy = np.concatenate([st0.values_to_numpy()[0:-2], st1.values_to_numpy()[0:-2]])
#             # log.log('corrected state', np.matmul(Hx, state_as_numpy))
#             log.log('--------------------------------------------\n')
#
#             if index1+1 < buffer_length:
#                 for i in range(index1+1, buffer_length):
#                     ist = self._state_buffer.get_state(i)
#                     self.predict(ist.accel_input, ist.accel_var, ist.angular_input, ist.angular_var, ist.covariance[9:12,9:12],
#                                  ist.covariance[12:15, 12:15], ist.state_time, i - 1, measurementname + '_correction @ ' + str(time1))
#
#     def backward_smooth(self, index, dx, unsmooth_cov):
#         for i in range(index, 0, -1):
#             st_prev = self._state_buffer.get_state(i-1)
#             st_curr = self._state_buffer.get_state(i)
#
#             p_prev = st_prev.position.value
#             v_prev = st_prev.velocity.value
#             q_prev = st_prev.orientation.value
#             ab_prev = st_prev.accel_bias.value
#             wb_prev = st_prev.angular_bias.value
#
#             am = st_prev.accel_input
#             wm = st_prev.angular_input
#
#             P_prev = st_prev.covariance
#             P_curr = st_curr.covariance
#
#             dt = st_curr.state_time - st_prev.state_time
#
#             tf_q = np.concatenate([q_prev[1:4], [q_prev[0]]])
#             R_inert_body = tft.quaternion_matrix(tf_q)[0:3, 0:3]
#
#             Fx = np.eye(15)
#             Fx[0:3, 3:6] = dt * np.eye(3)
#             Fx[3:6, 6:9] = - skew_symmetric(R_inert_body.dot(am - ab_prev)) * dt
#             Fx[3:6, 9:12] = -dt * R_inert_body
#             Fx[6:9, 12:15] = -dt * R_inert_body
#
#             if np.linalg.det(unsmooth_cov) == 0:
#                 log.log('non-invertible covariance matrix')
#                 log.log(unsmooth_cov)
#                 log.log('\n\n')
#
#             A = np.matmul(P_prev, np.matmul(Fx.T, np.linalg.inv(unsmooth_cov)))
#             dx = np.matmul(A, dx)
#             new_unsmooth_cov = deepcopy(P_prev)
#             P_prev = P_prev + np.matmul(A, np.matmul(P_curr - unsmooth_cov, A.T))
#             P_prev = 0.5 * (P_prev + P_prev.T)
#             unsmooth_cov = new_unsmooth_cov
#
#             p_prev = p_prev + dx[0:3]
#             v_prev = v_prev + dx[3:6]
#             # q_prev = Quaternion(axis_angle=dx[6:9]).quat_mult_left(q_prev, out='Quaternion').normalize().to_numpy()
#             angle = tft.vector_norm(dx[6:9])
#             axis = tft.unit_vector(dx[6:9])
#             q_prev = quaternion_xyzw2wxyz(
#                 tft.quaternion_multiply(tft.quaternion_about_axis(angle, axis), quaternion_wxyz2xyzw(q_prev)))
#             ab_prev = ab_prev + dx[9:12]
#             wb_prev = wb_prev + dx[12:15]
#
#             st_prev.position.value = p_prev
#             st_prev.velocity.value = v_prev
#             st_prev.orientation.value = q_prev
#             st_prev.accel_bias.value = ab_prev
#             st_prev.angular_bias.value = wb_prev
#             st_prev.covariance = P_prev
#             self._state_buffer.update_state(st_prev, i-1)
#
#
#     def get_state_as_numpy(self):
#         with self._lock:
#             return self._state_buffer.get_state(-1).values_to_numpy()
#
#
#     def is_initialized(self):
#         with self._lock:
#             return self._state_buffer.get_state(-1).initialized
#
#
#     def get_covariance_as_numpy(self):
#         with self._lock:
#             P = self._state_buffer.get_state(-1).covariance
#             return P.flatten()
