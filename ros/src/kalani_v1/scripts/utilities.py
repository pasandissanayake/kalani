from __future__ import print_function
import __builtin__
import yaml
import numpy as np
import time
from scipy.optimize import leastsq

import tf.transformations as tft


class Log:
    '''
        class to handle logging and printing
    '''
    def __init__(self, prefix):
        self._prefix = str(prefix)

    def log(self, *args, **kwargs):
        __builtin__.print(self._prefix, end=' := ')
        return __builtin__.print(*args, **kwargs)


class Stopwatch:
    '''
        class to measure timing
    '''
    def __init__(self):
        self._start_time = -1
        self._stopped = True
        self._log = Log('Stopwatch')

    def start(self):
        if self._stopped:
            self._start_time = time.time()
            self._stopped = False
        else:
            self._log.log('Already running.')

    def lap(self):
        if self._stopped:
            self._log.log('Stopwatch has stopped.')
            return 0
        else:
            return time.time() - self._start_time

    def stop(self):
        if self._stopped:
            self._log.log('Stopwatch has stopped.')
            return_val = 0
        else:
            return_val = time.time() - self._start_time
        del self
        return return_val


def get_config_dict():
    '''
    :returns: dictionary containing configuration values, loaded from config.yaml file
    '''
    with open('/home/pasan/kalani/ros/src/kalani_v1/scripts/config.yaml') as f:
        return yaml.load(f, Loader=yaml.Loader)


def get_orientation_from_magnetic_field(mm, fm):
    '''
    method for estimating orientation using accelerometer and magnetometer data
    :param mm: 3x1 numpy array of magnetic field - [left, forward, up]
    :param fm: 3x1 numpy array of accelerations - [left, forward, up]
    :return: unit quaternion respect to the ENU frame - [x, y, z, w]
    '''
    mm = np.array(mm)
    fm = np.array(fm)

    me = np.array([0, -1, 0])
    ge = np.array([0, 0, -1])

    fs = fm / np.linalg.norm(fm)
    ms = mm - np.dot(mm, fs) * fs
    ms = ms / np.linalg.norm(ms)

    def eqn(p):
        x, y, z, w = p
        R = tft.quaternion_matrix([x,y,z,w])[0:3,0:3]
        M = R.dot(ms) + me
        G = R.dot(fs) + ge
        E = np.concatenate((M,G,[w**2+x**2+y**2+z**2-1]))
        return E
    x, y, z, w = leastsq(eqn, tft.quaternion_from_euler(0,0,np.pi/4,axes='sxyz'))[0]

    quat_array = tft.unit_vector([x, y, z, w])
    if quat_array[-1] < 0: quat_array = -quat_array

    return quat_array


def get_position_from_gnss_fix(fix, origin, fixunit='deg', originunit='deg'):
    '''
    convert gnss coordinates to local coordinates in ENU frame
    :param fix: gnss coordinate / coordinate array - [lat, lon]
    :param origin: gnss coordinate of the origin - [lat, lon]
    :param fixunit: units of fix ('deg'-->degrees, 'rad'-->radians)
    :param originunit: units of origin ('deg'-->degrees, 'rad'-->radians)
    :return: coordinates in local ENU frame
    '''
    if fixunit == 'deg':
        fix = np.deg2rad(fix)
    else:
        fix = np.array(fix)

    if originunit == 'deg':
        origin = np.deg2rad(origin)
    else:
        origin = np.array(origin)

    r = 6400000
    dif = fix - origin
    if np.ndim(fix) == 2:
        x = r * np.cos(origin[0]) * np.sin(dif[:, 1])
        y = r * np.sin(dif[:, 0])
    else:
        x = r * np.cos(origin[0]) * np.sin(dif[1])
        y = r * np.sin(dif[0])

    return np.array([x,y]).T


def quaternion_xyzw2wxyz(q):
    '''
    converts quaterenion in xyzw form to wxyz form
    :param q: quaternion (4x1) / array of quaternions (nx4) in xyzw
    :return: converted quaternion/ array of quaternions in wxyz form
    '''
    q = np.array(q)
    dims = np.ndim(q)

    # single quaternion
    if dims == 1:
        return np.concatenate([[q[3]], q[:3]])

    # array of quaternions
    elif dims == 2:
        return np.concatenate([[q[:, 3]], q[:, :3].T]).T


def quaternion_wxyz2xyzw(q):
    '''
        converts quaterenion in wxyzw form to xyzw form
        :param q: quaternion (4x1) / array of quaternions (nx4) in wxyz
        :return: converted quaternion/ array of quaternions in xyzw form
    '''
    q = np.array(q)
    dims = np.ndim(q)

    # single quaternion
    if dims == 1:
        return np.concatenate([q[1:4], [q[0]]])

    # array of quaternions
    elif dims == 2:
        return np.concatenate([q[:, 1:4].T, [q[:, 0]]]).T


def skew_symmetric(v):
    '''
    skew-symmetric matrix of a vector
    :param v: 3x1 vector
    :return: 3x3 skew-symmetri matrix
    '''
    return np.array(
        [[    0, -v[2],  v[1]],
         [ v[2],     0, -v[0]],
         [-v[1],  v[0],    0]], dtype=np.float64)
