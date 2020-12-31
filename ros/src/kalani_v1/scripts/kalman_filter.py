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
        # self._buffer = [StateObject()]


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

        template_len = lambda t: sum([s[1] for s in t])
        self._ns_len = template_len(ns_template)
        self._es_len = template_len(es_template)
        self._pn_len = template_len(pn_template)
        self._mmi_len = template_len(mmi_template)

        self._ns_timestamps = {s[0]:-1 for s in ns_template}

        def ns_motion_model(ns, mmi, dt):
            pn = sarray(self._pn_template, np.zeros(self._pn_len))
            return motion_model(ns, mmi, pn, dt)
        self._motion_model = ns_motion_model
        self._combination = combination
        self._difference = difference

        def true_state(es, ns):
            ns = sarray(ns_template, ns)
            es = sarray(es_template, es)
            return combination(ns, es)
        def fx(es, ns, pn, mmi, dt):
            pn = sarray(pn_template, pn)
            pn_zero = sarray(pn_template, np.zeros(self._pn_len))
            mmi = sarray(mmi_template, mmi)
            ts = sarray(ns_template, true_state(es, ns))
            ns = sarray(ns_template, ns)
            mm_full = sarray(ns_template, motion_model(ts, mmi, pn, dt))
            mm_nill = sarray(ns_template, motion_model(ns, mmi, pn_zero, dt))
            return difference(mm_full, mm_nill)
        def fi(pn, es, ns, mmi, dt):
            return fx(es, ns, pn, mmi, dt)
        self._true_state_jacob = nd.Jacobian(true_state)
        self._Fx_jacob = nd.Jacobian(fx)
        self._Fi_jacob = nd.Jacobian(fi)


    def initialize(self, vals):
        # check whether a state object is available within the buffer
        if self._state_buffer.get_buffer_length() > 0:
            so = self._state_buffer.get_state(-1)
            update = True
        else:
            so = StateObject(self._ns_template, self._es_template, self._mmi_template)
            update = False
        key_min = min(self._ns_timestamps.keys(), key=(lambda k: self._ns_timestamps[k]))
        val_min = self._ns_timestamps[key_min]
        for val in vals:
            # reject if timestamp is older than the current oldest timestamp
            if val[3] < val_min:
                continue
            # set nominal state value
            fun_val = getattr(so.predicted_ns, val[0])
            fun_val()[:] = np.array(val[1])
            # set corresponding error state covariance
            fun_cov = getattr(so.predicted_es_cov, val[0])
            fun_cov()[:] = np.array(val[2])
            self._ns_timestamps[val[0]] = val[3]
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


    def predict(self, mmi_in, mmi_cov_in, timestamp, load_index=-1, input_name='unspecified'):
        with self._lock:
            mmi = sarray(self._mmi_template, mmi_in)
            mmi_cov = np.array(mmi_cov_in)

            so = self._state_buffer.get_state(load_index)
            if so is None or not so.is_valid:
                print('state not initialized')
                return
            so_time = so.timestamp
            if so_time >= timestamp:
                log.log('MM input {} is too old. Input time: {} s, closest state time: {} s, loadindex: {}'.format(input_name, timestamp, so_time, load_index))
                return

            dt = timestamp - so_time

            Fx = np.array(self._Fx_jacob(np.zeros(self._es_len), so.predicted_ns, np.zeros(self._pn_len), mmi, dt))
            Fi = np.array(self._Fi_jacob(np.zeros(self._pn_len), np.zeros(self._es_len), so.predicted_ns, mmi, dt))
            predicted_P = Fx.dot(so.predicted_es_cov).dot(Fx.T) + Fi.dot(mmi_cov).dot(Fi.T)

            predicted_ns = self._motion_model(so.predicted_ns, mmi, dt)

            print predicted_ns

            so_new = StateObject(self._ns_template, self._es_template, self._mmi_template)
            so_new.ns = sarray(self._ns_template, predicted_ns)
            so_new.es = sarray(self._es_template, np.zeros(self._es_len))
            so_new.es_cov = parray(self._es_template, predicted_P)
            so_new.predicted_ns = sarray(self._ns_template, predicted_ns)
            so_new.predicted_es = sarray(self._es_template, np.zeros(self._es_len))
            so_new.predicted_es_cov = parray(self._es_template, predicted_P)
            so_new.mm_inputs = sarray(self._mmi_template, mmi)
            so_new.mm_inputs_cov = parray(self._pn_template, mmi_cov)
            so_new.Fx = Fx
            so_new.Fi = Fi
            so_new.timestamp = timestamp
            so_new.is_valid = True

            if load_index == -1:
                self._state_buffer.add_state(so_new)
            else:
                self._state_buffer.update_state(so_new, load_index + 1)


    def correct_absolute(self, meas_fun, meas_cov, timestamp, hx_fun=None, measurement_name='unspecified'):
        with self._lock:
            oldest_ts = self._state_buffer.get_state(0).timestamp
            latest_ts = self._state_buffer.get_state(-1).timestamp
            if timestamp < oldest_ts:
                log.log(
                    'Measurement {} is too early. Measurement time: {}, latest state time: {}'.format(measurement_name,
                                                                                                        timestamp,
                                                                                                        latest_ts))
                return
            elif timestamp > latest_ts + self.STATE_INIT_TIME_THRESHOLD:
                log.log(
                    'Measurement {} is too old. Measurement time: {}, oldest state time: {}'.format(measurement_name,
                                                                                                    timestamp,
                                                                                                    oldest_ts))
                return

            so_index = self._state_buffer.get_index_of_closest_state_in_time(timestamp)
            so = self._state_buffer.get_state(so_index)

            X = self._true_state_jacob(so.es, so.ns)
            if hx_fun is not None:
                Hx = hx_fun(so.ns)
            else:
                def meas_fun_dup(ns):
                    ns = sarray(self._ns_template, ns)
                    return meas_fun(ns)
                Hx = nd.Jacobian(meas_fun_dup)(so.ns)
            H = np.matmul(Hx, X)

            P = so.es_cov
            K = np.matmul(np.matmul(P, H.T), np.linalg.inv(np.matmul(np.matmul(H, P), H.T) + meas_cov))
            corrected_P = np.matmul(np.eye(15) - np.matmul(K, H), P)
            corrected_P = 0.5 * (corrected_P + corrected_P.T)

            dx = sarray(self._es_template, K.dot(meas_fun(so.ns)))

            corrected_ns = self._combination(so.ns, dx)

            print corrected_ns

            so_new = StateObject(self._ns_template, self._es_template, self._mmi_template)
            so_new.ns = sarray(self._ns_template, corrected_ns)
            so_new.es = sarray(self._es_template, np.zeros(self._es_len))
            so_new.es_cov = parray(self._es_template, corrected_P)
            so_new.predicted_ns = sarray(self._ns_template, so.predicted_ns)
            so_new.predicted_es = sarray(self._es_template, np.zeros(self._es_len))
            so_new.predicted_es_cov = parray(self._es_template, so.predicted_es_cov)
            so_new.mm_inputs = sarray(self._mmi_template, so.mm_inputs)
            so_new.mm_inputs_cov = parray(self._pn_template, so.mm_inputs_cov)
            so_new.Fx = so.Fx
            so_new.Fi = so.Fi
            so_new.timestamp = so.timestamp
            so_new.is_valid = True










# motion model
def motion_model(ts, mmi, pn, dt):
    gravity = np.array([0, 0, -9.8])
    R = tft.quaternion_matrix(ts.q())[0:3,0:3]
    acc = (np.matmul(R, (mmi.a()-ts.ab())) + gravity)

    dtheta = (mmi.w() - ts.wb()) * dt + pn.q()
    angle, axis = axisangle_to_angle_axis(dtheta)
    dq = tft.quaternion_about_axis(angle, axis)

    res = [
        ts.p() + ts.v() * dt + 0.5 * acc * dt**2,
        ts.v() + acc * dt + pn.v(),
        tft.quaternion_multiply(ts.q(), dq),
        ts.ab() + pn.ab(),
        ts.wb() + pn.wb()
    ]
    return np.concatenate(res)


# nominal state, error state conjugation
# nominal state + error state ==> nominal state
def combination(ns, es):
    angle, axis = axisangle_to_angle_axis(es.q())
    res = [
        ns.p() + es.p(),
        ns.v() + es.v(),
        tft.quaternion_multiply(ns.q(), tft.quaternion_about_axis(angle, axis)),
        ns.ab() + es.ab(),
        ns.wb() + es.wb()
    ]
    return np.concatenate(res)


# nominal state difference
# nominal state(ns1) - nominal state(ns0) ==> error state
def difference(ns1, ns0):
    angle, axis = quaternion_to_angle_axis(tft.quaternion_multiply(tft.quaternion_inverse(ns0.q()), ns1.q()))
    res = [
        ns1.p() - ns0.p(),
        ns1.v() - ns0.v(),
        axis * angle,
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
    ['p', [1, 2, 3], [[1,0,0],[0,1,0],[0,0,1]], 1.7]
])

kf.initialize([
    ['q', [0, 0, 0, 1], [[1,0,0],[0,1,0],[0,0,1]], 1.7]
])

kf.initialize([
    ['v',  [3,3,3], np.eye(3), 2.0],
    ['ab', [4,4,4], np.eye(3), 2.0],
    ['wb', [2,2,2], np.eye(3), 2.0]
])


kf.predict(np.concatenate([[1,1,1],[2,2,2]]), np.eye(12), 2.5)

def meas_fun(ns):
    return np.ones(3) - ns.p()

def hx_fun(ns):
    return nd.Jacobian(meas_fun)(ns)

kf.correct_absolute(meas_fun, np.eye(3), 2.5, measurement_name='apple')