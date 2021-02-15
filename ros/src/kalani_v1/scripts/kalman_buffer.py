import numpy as np
import threading
from copy import deepcopy

from utilities import *

log = Log('kalman_buffer')

class ValueTimePair:
    def __init__(self, value, time):
        self.value = value
        self.time = time

class KalmanStateObject:
    def __init__(self):
        self.position = ValueTimePair(np.zeros(3), -1)
        self.velocity = ValueTimePair(np.zeros(3), -1)
        self.orientation = ValueTimePair(np.array([1,0,0,0]), -1)
        self.accel_bias = ValueTimePair(np.zeros(3), -1)
        self.angular_bias = ValueTimePair(np.zeros(3), -1)

        self.covariance = np.zeros((15,15))

        self.accel_input = np.zeros(3)
        self.accel_var = 0
        self.angular_input = np.zeros(3)
        self.angular_var = 0

        self.state_time = -1
        self.initialized = False

    def values_to_numpy(self):
        return np.concatenate(
            [self.position.value, self.velocity.value, self.orientation.value, self.accel_bias.value,
             self.angular_bias.value, [self.state_time, float(self.initialized)]])

    def times_to_numpy(self):
        return np.array([self.position.time, self.velocity.time, self.orientation.time,
                      self.accel_bias.time, self.angular_bias.time])


class KalmanBuffer:
    def __init__(self, bufferlength):
        self._lock = threading.Lock()

        # no. of previous states stored
        self._BUFFER_LENGTH = bufferlength

        # buffer for storing state variables. lowest index for oldest state
        self._buffer = [KalmanStateObject()]


    def add_state(self):
        with self._lock:
            so = KalmanStateObject()
            self._buffer.append(so)
            if len(self._buffer) > self._BUFFER_LENGTH:
                self._buffer.pop(0)


    def update_state(self, stateobject, index=-1):
        with self._lock:
            bl = len(self._buffer)
            if index >= bl or index < -bl:
                log.log('Store index: ' + str(index) + ' is out of range.')
            else:
                self._buffer[index] = deepcopy(stateobject)


    def get_state(self, index):
        with self._lock:
            bl = len(self._buffer)
            if index >= bl or index < -bl:
                log.log('Store index: ' + str(index) + ' is out of range.')
                return None
            else:
                return deepcopy(self._buffer[index])


    def get_timestamp_of_oldest_variable(self, index):
        state = self.get_state(index)
        timestamps = state.times_to_numpy()
        index = min(range(len(timestamps)), key=lambda i: timestamps[i])
        return timestamps[index]


    def get_timestamp_of_newest_variable(self, index):
        state = self.get_state(index)
        timestamps = state.times_to_numpy()
        index = max(range(len(timestamps)), key=lambda i: timestamps[i])
        return timestamps[index]


    def get_index_of_closest_state_in_time(self, time):
        with self._lock:
            state_timestamps = [st.state_time for st in self._buffer]
            index = min(range(len(state_timestamps)), key=lambda i: abs(state_timestamps[i] - time))
            return index


    def get_buffer_length(self):
        with self._lock:
            return len(self._buffer)


    def get_states_as_numpy(self):
        states = [self.get_state(i).values_to_numpy() for i in range(self.get_buffer_length())]
        return np.array(states)