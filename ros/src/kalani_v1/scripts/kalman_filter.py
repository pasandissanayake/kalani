import numpy as np
import sympy as sp
import numdifftools as nd
import threading
from copy import deepcopy, copy
from types import MethodType
import gnumpy as gnp

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
    def __init__(self, ns_template, es_template, pn_template, mmi_template, motion_model, combination, difference, ts_fun=None, fx_fun=None, fi_fun=None):
        # State buffer length
        self.STATE_BUFFER_LENGTH = kf_config['buffer_size']

        # Allowed maximum gap between any two initial state variables (in seconds)
        self.STATE_INIT_TIME_THRESHOLD = kf_config['state_init_time_threshold']

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
        self.is_initialized = False

        self._last_corrected_index = 0

        def ns_motion_model(ns, mmi, dt):
            pn = sarray(self._pn_template, np.zeros(self._pn_len))
            return motion_model(ns, mmi, pn, dt)
        def ns_combination(ns, es):
            ns = sarray(self._ns_template, ns)
            es = sarray(self._es_template, es)
            return combination(ns, es)
        def ns_difference(ns1, ns0):
            ns1 = sarray(self._ns_template, ns1)
            ns0 = sarray(self._ns_template, ns0)
            return difference(ns1, ns0)
        self._motion_model = ns_motion_model
        self._combination = ns_combination
        self._difference = ns_difference

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

        if ts_fun is None:
            self._true_state_jacob = nd.Jacobian(true_state)
        else:
            def ts_jacob(es, ns):
                ns = sarray(ns_template, ns)
                es = sarray(es_template, es)
                return ts_fun(ns)
            self._true_state_jacob = ts_jacob
        if fx_fun is None:
            self._Fx_jacob = nd.Jacobian(fx)
        else:
            def fx_jacob(es, ns, pn, mmi, dt):
                mmi = sarray(mmi_template, mmi)
                ns = sarray(ns_template, ns)
                return fx_fun(ns, mmi, dt)
            self._Fx_jacob = fx_jacob
        if fi_fun is None:
            self._Fi_jacob = nd.Jacobian(fi)
        else:
            def fi_jacob(pn, es, ns, mmi, dt):
                mmi = sarray(mmi_template, mmi)
                ns = sarray(ns_template, ns)
                return fi_fun(ns, mmi, dt)
            self._Fi_jacob = fi_jacob


    def initialize(self, vals):
        # vals in the form ['sub state name', value, cov, timestamp]
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
            fun_val = getattr(so.ns, val[0])
            fun_val()[:] = np.array(val[1])
            # set corresponding error state covariance
            fun_cov = getattr(so.es_cov, val[0])
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
        self.is_initialized = is_initialized
        if self.is_initialized:
            log.log('Filter initialized.')
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
            if so_time > timestamp:
                log.log('MM input "{}" is too old. Input time: {} s, closest state time: {} s, loadindex: {}'.format(input_name, timestamp, so_time, load_index))
                return

            dt = timestamp - so_time

            Fx = np.array(self._Fx_jacob(np.zeros(self._es_len), so.ns, np.zeros(self._pn_len), mmi, dt))
            Fi = np.array(self._Fi_jacob(np.zeros(self._pn_len), np.zeros(self._es_len), so.ns, mmi, dt))

            predicted_P = Fx.dot(so.es_cov).dot(Fx.T) + Fi.dot(mmi_cov).dot(Fi.T)

            predicted_ns = self._motion_model(so.ns, mmi, dt)

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
                if self._last_corrected_index > 0:
                    self._last_corrected_index = self._last_corrected_index - 1
            else:
                self._state_buffer.update_state(so_new, load_index + 1)
                    

    def correct_absolute(self, meas_fun, measurement, meas_cov, timestamp, hx_fun=None, constraints=None, measurement_name='unspecified'):
        with self._lock:
            oldest_ts = self._state_buffer.get_state(0).timestamp
            latest_ts = self._state_buffer.get_state(-1).timestamp
            if timestamp < oldest_ts:
                log.log(
                    'Measurement "{}" is too old. Measurement time: {}, oldest state time: {}'.format(measurement_name,
                                                                                                      timestamp,
                                                                                                      oldest_ts))
                return
            elif timestamp > latest_ts + self.STATE_INIT_TIME_THRESHOLD:
                log.log(
                    'Measurement "{}" is too early. Measurement time: {}, latest state time: {}'.format(
                        measurement_name,
                        timestamp,
                        latest_ts))
                return

            so_index = self._state_buffer.get_index_of_closest_state_in_time(timestamp)
            so = self._state_buffer.get_state(so_index)

            if so_index < self._last_corrected_index:
                log.log('bws in abs cor: from {} to {}'.format(self._last_corrected_index, so_index))
                swM = Stopwatch()
                swM.start()
                for j in range(self._last_corrected_index, so_index-1, -1):
                    self.backward_smooth(j)
                log.log('abs cor bw end in: {} s'.format(swM.stop()))
            
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
            corrected_P = np.matmul(np.eye(self._es_len) - np.matmul(K, H), P)
            corrected_P = 0.5 * (corrected_P + corrected_P.T)

            dx = sarray(self._es_template, np.matmul(K, measurement-meas_fun(so.ns)))

            if constraints is not None:
                dx = sarray(self._es_template, constraints(dx))

            log.log('type: {}  so index: {} last cor idx: {}'.format(measurement_name, so_index, self._last_corrected_index))
            # log.log('dx p:{}'.format(dx[0:2]))
            # log.log('H: \n{}'.format(H))

            corrected_ns = self._combination(so.ns, dx)

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

            self._state_buffer.update_state(so_new, so_index)

            self._last_corrected_index = so_index

            buffer_length = self._state_buffer.get_buffer_length()
            if so_index+1 < buffer_length:
                for i in range(so_index+1, buffer_length):
                    so_i = self._state_buffer.get_state(i)
                    self.predict(so_i.mm_inputs, so_i.mm_inputs_cov, so_i.timestamp, i - 1,
                                 measurement_name + '_correction @ ' + str(timestamp))
            
            # for j in range(so_index, 1, -1):
            #     self.backward_smooth(j)
                                    

    def correct_relative(self, meas_fun, measurement, measurement_cov, timestamp1, timestamp0, hx_fun=None, measurement_name='unspecified'):
        swB = Stopwatch()
        swB.start()
        oldest_ts = self._state_buffer.get_state(0).timestamp
        latest_ts = self._state_buffer.get_state(-1).timestamp
        if timestamp0 >= timestamp1:
            log.log('Measurement {} has inverted timestamps. timestamp0: {}, timestamp1: {}'.format(measurement_name, timestamp0, timestamp1))
            return
        elif timestamp0 < oldest_ts:
            log.log(
                'Measurement "{}" is too old. Measurement time: {}, oldest state time: {}'.format(measurement_name,
                                                                                                    timestamp0,
                                                                                                    oldest_ts))
            return
        elif timestamp1 > latest_ts + self.STATE_INIT_TIME_THRESHOLD:
            log.log(
                'Measurement "{}" is too early. Measurement time: {}, latest state time: {}'.format(
                    measurement_name,
                    timestamp1,
                    latest_ts))
            return
        
        so0_index = self._state_buffer.get_index_of_closest_state_in_time(timestamp0)
        so1_index = self._state_buffer.get_index_of_closest_state_in_time(timestamp1)
        swA = Stopwatch()
        swA.start()
        for j in range(so1_index, so0_index-2, -1):
            self.backward_smooth(j)
        log.log("rc wo bs: {} s, no of states: {}".format(swA.stop(),so1_index-so0_index))
        so0 = self._state_buffer.get_state(so0_index)
        so1 = self._state_buffer.get_state(so1_index)

        Fx_prod = np.eye(self._es_len)
        for i in range(so1_index, so0_index, -1):
            Fxi = self._state_buffer.get_state(i).Fx
            Fx_prod = np.matmul(Fx_prod, Fxi)
        
        P = np.eye(self._es_len * 2)
        P[0:self._es_len, 0:self._es_len] = so0.es_cov
        P[self._es_len:self._es_len*2, self._es_len:self._es_len*2] = so1.es_cov
        P[0:self._es_len, self._es_len:self._es_len*2] = np.matmul(so0.es_cov, Fx_prod.T)
        P[self._es_len:self._es_len*2, 0:self._es_len] = np.matmul(Fx_prod, so0.es_cov)

        X0 = self._true_state_jacob(so0.es, so0.ns)
        X1 = self._true_state_jacob(so1.es, so1.ns)
        
        if hx_fun is not None:
            Hx1, Hx0 = hx_fun(so1.ns, so0.ns)
        else:
            def meas_fun_dup1(ns1, ns0):
                ns0 = sarray(self._ns_template, ns0)
                ns1 = sarray(self._ns_template, ns1)
                return meas_fun(ns1, ns0)
            def meas_fun_dup0(ns0, ns1):
                ns0 = sarray(self._ns_template, ns0)
                ns1 = sarray(self._ns_template, ns1)
                return meas_fun(ns1, ns0)
            Hx1 = nd.Jacobian(meas_fun_dup1)(so1.ns, so0.ns)
            Hx0 = nd.Jacobian(meas_fun_dup0)(so0.ns, so1.ns)
                    
        H0 = np.matmul(Hx0, X0)
        H1 = np.matmul(Hx1, X1)

        H = np.concatenate([H0, H1], axis=1)

        S = np.matmul(np.matmul(H, P), H.T) + measurement_cov
        K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))

        K1 = K[self._es_len:self._es_len*2, :]
        corrected_P1 = so1.es_cov - np.matmul(np.matmul(K1, S), K1.T)
        dx1 = K1.dot(measurement - meas_fun(so1.ns, so0.ns))
        dx0 = K[0:self._es_len].dot(measurement - meas_fun(so1.ns, so0.ns))
        corrected_ns1 = self._combination(so1.ns, dx1)

        # so0an, so0ax = quaternion_to_angle_axis(so0.ns.q())
        # so0anax = so0an * so0ax
        # so1an, so1ax = quaternion_to_angle_axis(so1.ns.q())
        # so1anax = so1an * so1ax
        # log.log('so0_idx:{}, so1_idx:{}'.format(so0_index, so1_index))
        # log.log('so0:{}, so1:{}'.format(so0anax, so1anax))
        # log.log('meas:{}, meas_fun:{}, dif:{}'.format(measurement, meas_fun(so1.ns, so0.ns), measurement - meas_fun(so1.ns, so0.ns)))
        # log.log('dx0:{}'.format(dx0))
        # log.log('dx1:{}'.format(dx1))
        # log.log('H:{}'.format(H))

        so_new = StateObject(self._ns_template, self._es_template, self._mmi_template)
        so_new.ns = sarray(self._ns_template, corrected_ns1)
        so_new.es = sarray(self._es_template, np.zeros(self._es_len))
        so_new.es_cov = parray(self._es_template, corrected_P1)
        so_new.predicted_ns = sarray(self._ns_template, so1.predicted_ns)
        so_new.predicted_es = sarray(self._es_template, np.zeros(self._es_len))
        so_new.predicted_es_cov = parray(self._es_template, so1.predicted_es_cov)
        so_new.mm_inputs = sarray(self._mmi_template, so1.mm_inputs)
        so_new.mm_inputs_cov = parray(self._pn_template, so1.mm_inputs_cov)
        so_new.Fx = so1.Fx
        so_new.Fi = so1.Fi
        so_new.timestamp = so1.timestamp
        so_new.is_valid = True

        # prev_an, prev_ax = quaternion_to_angle_axis(so1.ns.q())
        # prev = prev_an * prev_ax
        # new_an, new_ax = quaternion_to_angle_axis(so_new.ns.q())
        # new = new_an * new_ax
        # log.log("prev:{}, new:{}".format(prev, new))

        self._state_buffer.update_state(so_new, so1_index)

        buffer_length = self._state_buffer.get_buffer_length()
        if so1_index + 1 < buffer_length:
            for i in range(so1_index + 1, buffer_length):
                so_i = self._state_buffer.get_state(i)
                self.predict(so_i.mm_inputs, so_i.mm_inputs_cov, so_i.timestamp, i - 1,
                             measurement_name + '_correction @ ' + str(timestamp1))
        # if so0_index > 0:
        #     for j in range(so1_index, so0_index, -1):
        #             self.backward_smooth(j)
        log.log('rc: {} s'.format(swB.stop()))

    def backward_smooth(self, load_index):
        if self._state_buffer.get_buffer_length() <= load_index or load_index <= 0:
            log.log('Backward smooth start index: "{}" is out of bounds'.format(load_index))
            return

        so_current = self._state_buffer.get_state(load_index)
        so_previous = self._state_buffer.get_state(load_index - 1)

        if not so_current.is_valid:
            log.log('Backward smooth current state (index: "{}") is invalid'.format(load_index))
            return

        if not so_previous.is_valid:
            log.log('Backward smooth previous state (index: "{}") is invalid'.format(load_index-1))
            return
        
        S = gnp.dot(so_previous.es_cov, gnp.dot(so_previous.Fx.T, np.linalg.inv(so_current.predicted_es_cov))).as_numpy_array()
        smoothed_ns = self._combination(so_previous.ns, gnp.dot(S, self._difference(so_current.ns, so_current.predicted_ns)).as_numpy_array())
        smoothed_P = so_previous.es_cov + gnp.dot(S, gnp.dot(so_current.es_cov - so_current.predicted_es_cov, S.T)).as_numpy_array()
        
        so_new = StateObject(self._ns_template, self._es_template, self._mmi_template)
        so_new.ns = sarray(self._ns_template, smoothed_ns)
        so_new.es = sarray(self._es_template, np.zeros(self._es_len))
        so_new.es_cov = parray(self._es_template, smoothed_P)
        so_new.predicted_ns = sarray(self._ns_template, so_previous.predicted_ns)
        so_new.predicted_es = sarray(self._es_template, np.zeros(self._es_len))
        so_new.predicted_es_cov = parray(self._es_template, so_previous.predicted_es_cov)
        so_new.mm_inputs = sarray(self._mmi_template, so_previous.mm_inputs)
        so_new.mm_inputs_cov = parray(self._pn_template, so_previous.mm_inputs_cov)
        so_new.Fx = so_previous.Fx
        so_new.Fi = so_previous.Fi
        so_new.timestamp = so_previous.timestamp
        so_new.is_valid = True

        self._state_buffer.update_state(so_new, load_index - 1)
                

    def get_current_state(self):
        so = self._state_buffer.get_state(-1)
        return so.ns, so.timestamp, so.is_valid


    def get_current_cov(self):
        return self._state_buffer.get_state(-1).es_cov.ravel()


    def get_ns_length(self):
        return copy(self._ns_len)


    def get_es_length(self):
        return copy(self._es_len)


    def print_states(self, title):
        log.log("********* {} *********".format(title))
        for sosi in range(self._state_buffer.get_buffer_length()):
            sos = self._state_buffer.get_state(sosi)
            sosan, sosax = quaternion_to_angle_axis(sos.ns.q())
            sosanax = sosan * sosax
            log.log('{}: {}'.format(sosi, sosanax))

    
    def get_no_previous_states(self):
        return self._state_buffer.get_buffer_length()