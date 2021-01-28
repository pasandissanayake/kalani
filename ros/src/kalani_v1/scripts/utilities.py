from __future__ import print_function
import __builtin__
import yaml
import numpy as np
import time
from scipy.optimize import leastsq
from pyproj import Proj

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


def get_utm_from_gnss_fix(fix, origin, zone, hemisphere, fixunit='deg'):
    '''
        convert gnss coordinates to local coordinates in ENU frame
        :param fix: gnss coordinate / coordinate array - [lat, lon]
        :param origin: gnss coordinate of the origin - [lat, lon]
        :param fixunit: units of fix ('deg'-->degrees, 'rad'-->radians)
        :param originunit: units of origin ('deg'-->degrees, 'rad'-->radians)
        :return: coordinates in local ENU frame
    '''
    if fixunit == 'rad':
        fix = np.rad2deg(fix)
    else:
        fix = np.array(fix)

    origin = np.array(origin)
    proj = Proj("+proj=utm +zone={}, +{} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(zone, hemisphere))

    if np.ndim(fix) == 2:
        ret = np.zeros((len(fix), 2))
        for i in range(len(fix)):
            utm = np.array(proj(fix[i, 1], fix[i, 0]))
            ret[i, :] = utm - origin
    else:
        utm = np.array(proj(fix[1], fix[0]))
        ret = utm - origin

    return ret


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

def quaternion_to_angle_axis(q):
    '''
    converts a quaternion in xyzw form to angle, axis form
    angle = cos^-1(w) * 2
    axis = 2 * [x, y, z] / angle
    :param q: quaternion in xyzw form
    :return: angle, axis
    '''
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    half_angle = np.arccos(q[3])
    if half_angle != 0:
        axis = np.array(q[:3]) / half_angle
        angle = half_angle * 2
        return angle, axis
    else:
        return 1e-5, np.ones(3)


def axisangle_to_angle_axis(v):
    '''
    extracts angle and axis from a vector representing an orientation in axis-angle form
    :param v: vector representing the orientation
    :return: angle, axis
    '''
    angle = np.linalg.norm(v)
    if angle > 0:
        axis = np.array(v) / angle
    else:
        axis = np.zeros(3)
    return angle, axis


def rpy_jacobian_axis_angle(a):
    """Jacobian of RPY Euler angles with respect to axis-angle vector."""
    if not (type(a) == np.ndarray and len(a) == 3):
        raise ValueError("'a' must be a np.ndarray with length 3.")
    # From three-parameter representation, compute u and theta.
    na = np.sqrt(a.dot(a))
    na3 = na**3
    t = np.sqrt(a.dot(a))
    u = a/t

    # First-order approximation of Jacobian wrt u, t.
    Jr = np.array([[t/(t**2*u[0]**2 + 1), 0, 0, u[0]/(t**2*u[0]**2 + 1)],
                   [0, t/np.sqrt(1 - t**2*u[1]**2), 0, u[1]/np.sqrt(1 - t**2*u[1]**2)],
                   [0, 0, t/(t**2*u[2]**2 + 1), u[2]/(t**2*u[2]**2 + 1)]])

    # Jacobian of u, t wrt a.
    Ja = np.array([[(a[1]**2 + a[2]**2)/na3,        -(a[0]*a[1])/na3,        -(a[0]*a[2])/na3],
                   [       -(a[0]*a[1])/na3, (a[0]**2 + a[2]**2)/na3,        -(a[1]*a[2])/na3],
                   [       -(a[0]*a[2])/na3,        -(a[1]*a[2])/na3, (a[0]**2 + a[1]**2)/na3],
                   [                a[0]/na,                 a[1]/na,                 a[2]/na]])

    return Jr.dot(Ja)