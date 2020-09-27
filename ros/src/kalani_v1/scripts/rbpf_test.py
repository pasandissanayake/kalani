import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal
import tf.transformations as tft
from datasetutils.nclt_data_conversions import *

##################################################################
#  quaternion representation:
#      q = (x, y, z, w)
##################################################################

def get_orientation_from_magnetic_field(mm, fm):
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
    x, y, z, w = scipy.optimize.leastsq(eqn, tft.quaternion_from_euler(0,0,np.pi/4,axes='sxyz'))[0]

    quat_array = tft.unit_vector([x, y, z, w])
    if quat_array[0] < 0: quat_array = -quat_array

    return quat_array


def quaternion_multiply(q1, q0):
    results = []
    for i in range(len(q0)):
        results.append(tft.quaternion_multiply(q1[i],q0[i]))
    return np.array(results)

def quaternion_matrix(q):
    results = []
    for i in range(len(q)):
        results.append(tft.quaternion_matrix(q[i])[0:3,0:3])
    return np.array(results)

def quaternion_from_euler(r, p, y):
    results = []
    for i in range(len(r)):
        results.append(tft.quaternion_from_euler(r[i], p[i], y[i]))
    return np.array(results)

def euler_from_quaternion(q):
    results = []
    for i in range(len(q)):
        results.append(tft.euler_from_quaternion(q[i]))
    return np.array(results)

def quaternion_about_axis(angle, axis):
    results = []
    for i in range(len(angle)):
        results.append(tft.quaternion_about_axis(angle[i], axis[i]))
    return np.array(results)

def matrix_vector_multiply(matrix, vector):
    if np.ndim(matrix) > 2:
        if np.ndim(vector) > 1:
            return np.einsum('lij,lj->li', matrix, vector)
        else:
            return np.einsum('lij,j->li', matrix, vector)
    else:
        if np.ndim(vector) > 1:
            return np.einsum('ij,lj->li', matrix, vector)
        else:
            return np.einsum('ij,j->i', matrix, vector)

def matrix_matrix_multiply(m1, m2):
    if np.ndim(m1) > 2:
        if np.ndim(m2) > 2:
            return np.einsum('lij,ljk->lik', m1, m2)
        else:
            return np.einsum('lij,jk->lik', m1, m2)
    else:
        if np.ndim(m2) > 2:
            return np.einsum('ij,ljk->lik', m1, m2)
        else:
            return np.einsum('ij,jk->ik', m1, m2)

def matrix_transpose(matrix):
    if np.ndim(matrix) > 2:
        return np.einsum('lij->lji', matrix)
    else:
        return matrix.T

def matrix_inverse(matrix):
    results = []
    if np.ndim(matrix) > 2:
        for i in range(len(matrix)):
            results.append(np.linalg.inv(matrix[i]))
        return results
    else:
        return np.linalg.inv(matrix)

def multivariate_normal_pdf(points, means, covariances):
    if np.ndim(points) > 1:
        results = []
        for i in range(len(points)):
            results.append(multivariate_normal.pdf(points[i], mean=means[i], cov=covariances[i]))
        return np.array(results)
    elif np.ndim(means) > 1:
        results = []
        for i in range(len(means)):
            results.append(multivariate_normal.pdf(points, mean=means[i], cov=covariances[i]))
        return np.array(results)
    else:
        return multivariate_normal.pdf(points, mean=means, cov=covariances)


class RB_Particle_Filter_V1():
    def __init__(self):
        self.NO_OF_PARTICLES = 100 * 3

        self._pf_state = np.zeros([self.NO_OF_PARTICLES, 4])
        self._pf_weights = np.ones(self.NO_OF_PARTICLES) * 1 / self.NO_OF_PARTICLES
        self._kf_state = np.zeros([self.NO_OF_PARTICLES, 9])
        self._kf_covariance = np.zeros([self.NO_OF_PARTICLES, 9, 9])
        self._filter_time = -1
        self._mean_state = np.zeros(13)
        self._mean_covariance = np.zeros((9, 9))
        self._mean_filter_time = -1
        self._filter_initialized = False
        self._g = np.zeros(3)


    def initialize_state(self, p=None, cov_p=None, v=None, cov_v=None, q=None, cov_q=None, ab=None, cov_ab=None, g=None, time=-1, initialized=None):
        if p is not None:
            if cov_p is not None:
                self._kf_state[:, 0:3] = np.array(p)
                self._kf_covariance[:, 0:3, 0:3] = np.diag(cov_p)
            else:
                print('position covariances are not provided')

        if v is not None:
            if cov_v is not None:
                self._kf_state[:, 3:6] = np.array(v)
                self._kf_covariance[:, 3:6, 3:6] = np.diag(cov_v)
            else:
                print('velocity covariances are not provided')

        if q is not None:
            if cov_q is not None:
                euler = tft.euler_from_quaternion(q)
                # limit = 0.0001 * self.NO_OF_PARTICLES / 3
                # noise = np.random.uniform(-limit, limit, (np.round(self.NO_OF_PARTICLES), 3))
                noise = np.random.multivariate_normal(np.zeros(3), np.diag(cov_q), self.NO_OF_PARTICLES)
                neulers = noise + euler
                self._pf_state = quaternion_from_euler(neulers[:, 0], neulers[:, 1], neulers[:, 2])
                self._pf_weights = np.ones(self.NO_OF_PARTICLES) / self.NO_OF_PARTICLES
            else:
                print('orientation covariances are not provided')

        if ab is not None:
            if cov_ab is not None:
                self._kf_state[:, 6:9] = np.array(ab)
                self._kf_covariance[:, 6:9, 6:9] = np.diag(cov_ab)
            else:
                print('acceleration bias covariances are not provided')

        if g is not None:
            self._g = g

        if initialized is not None:
            self._filter_initialized = initialized

        self._filter_time = time
        self._mean_filter_time = time

        if self._filter_initialized == True:
            print('state initialized')


    def predict(self, am, var_am, wm, time):
        dt = time - self._filter_time

        if dt <= 0:
            print 'old input!'
            return

        R = quaternion_matrix(self._pf_state)

        angle = np.linalg.norm(wm * dt)
        axis = wm * dt / angle
        dq = tft.quaternion_about_axis(angle, axis)
        dq = np.full((self.NO_OF_PARTICLES, 4), dq)
        self._pf_state = quaternion_multiply(self._pf_state, dq)

        self._kf_state[:, 0:3] = self._kf_state[:, 0:3] + self._kf_state[:, 3:6] * dt + 0.5 * (matrix_vector_multiply(R, am - self._kf_state[:, 6:9]) + self._g) * dt ** 2
        self._kf_state[:, 3:6] = self._kf_state[:, 3:6] + (matrix_vector_multiply(R, am - self._kf_state[:, 6:9]) + self._g) * dt
        self._kf_state[:, 6:9] = self._kf_state[:, 6:9]

        F = np.full((self.NO_OF_PARTICLES,9,9),np.eye(9))
        F[:, 0:3, 3:6] = dt
        F[:, 0:3, 6:9] = -0.5 * dt**2 * R
        F[:, 3:6, 6:9] = -dt * R

        B = np.zeros((self.NO_OF_PARTICLES, 9, 3))
        B[:, 0:3, 0:3] = 0.5 * dt**2 * R
        B[:, 3:6, 0:3] = dt * R

        Q = np.diag(np.ones(3) * var_am)

        self._kf_covariance = matrix_matrix_multiply(F, matrix_matrix_multiply(self._kf_covariance, matrix_transpose(F))) + matrix_matrix_multiply(B, matrix_matrix_multiply(Q, matrix_transpose(B)))

        self.set_mean_state()


    def correct(self, fix, var_fix, time):
        if abs(time - self._filter_time) > 1:
            print 'old correction!'
            return

        H = np.zeros((3,9))
        H[0:3, 0:3] = np.eye(3)

        y = fix - matrix_vector_multiply(H, self._kf_state)
        S = matrix_matrix_multiply(H, matrix_matrix_multiply(self._kf_covariance, H.T)) + var_fix
        K = matrix_matrix_multiply(self._kf_covariance, matrix_matrix_multiply(H.T, matrix_inverse(S)))
        self._kf_covariance = matrix_matrix_multiply(np.eye(9) - matrix_matrix_multiply(K, H), self._kf_covariance)
        self._kf_covariance = 0.5 * (self._kf_covariance + matrix_transpose(self._kf_covariance))
        self._kf_state = self._kf_state + matrix_vector_multiply(K, y)

        pdf = multivariate_normal_pdf(fix, fix - y, S)
        if np.max(self._pf_weights * pdf) < 1e-10:
            print 'resampling...'
            nsamples = np.round(self._pf_weights * self.NO_OF_PARTICLES / np.sum(self._pf_weights))
            # print np.sum(nsamples)
            if np.sum(nsamples) != self.NO_OF_PARTICLES:
                deficit = self.NO_OF_PARTICLES - np.sum(nsamples)
                i = np.argmax(nsamples)
                nsamples[i] = nsamples[i] + deficit
            new_pf_state = []
            new_kf_state = []
            new_kf_covariance = []
            for i in range(len(nsamples)):
                for j in range(int(nsamples[i])):
                    new_pf_state.append(self._pf_state[i])
                    new_kf_state.append(self._kf_state[i])
                    new_kf_covariance.append(self._kf_covariance[i])
            self._pf_state = np.array(new_pf_state)

            euler = euler_from_quaternion(self._pf_state)
            noise = np.random.multivariate_normal(np.zeros(3), np.diag([0.1, 0.1, 1]), self.NO_OF_PARTICLES)
            neulers = noise + euler
            self._pf_state = quaternion_from_euler(neulers[:, 0], neulers[:, 1], neulers[:, 2])

            self._pf_weights = np.ones(self.NO_OF_PARTICLES) / self.NO_OF_PARTICLES
            self._kf_state = np.array(new_kf_state)
            self._kf_covariance = np.array(new_kf_covariance)
        else:
            print 'updating weights...'
            self._pf_weights = self._pf_weights * pdf
            self._pf_weights = self._pf_weights / np.sum(self._pf_weights)

        self.set_mean_state()



    def set_mean_state(self):
        d = np.zeros((4, 4))
        p = np.zeros((9, 9))
        x = np.zeros(9)
        for i in range(len(self._pf_weights)):
            d = d + self._pf_weights[i] * (np.outer(self._pf_state[i], self._pf_state[i]) - np.eye(4))
            p = p + self._pf_weights[i] * self._kf_covariance[i]
            x = x + self._pf_weights[i] * self._kf_state[i]

        vals, vecs = np.linalg.eig(d)
        index = np.argmax(vals)
        q = vecs[:, index]

        self._mean_state = np.concatenate([x, q])
        self._mean_covariance = p
        self._mean_filter_time = self._filter_time


    def get_state_as_numpy(self):
        return np.concatenate([self._mean_state[0:6], self._mean_state[9:13], self._mean_state[6:9], [0,0,0, self._mean_filter_time, float(self._filter_initialized)]])



pf = RB_Particle_Filter_V1()

nclt = NCLTData('/home/pasan/kalani/data/nclt/2012-04-29')

imur = np.loadtxt('/home/pasan/kalani/data/nclt/2012-04-29/ms25.csv', delimiter=',')
imu = np.zeros((len(imur), 10))
imu[:, 0] = imur[:, 0] * 1e-6
for j in range(3):
    i = j * 3
    imu[:, 1 + i] = imur[:, 2 + i]
    imu[:, 2 + i] = imur[:, 1 + i]
    imu[:, 3 + i] = -imur[:, 3 + i]

i_gps = 0
i_imu = 0
imu_ones = False
state = []
for i in range(200):
    t_gps = nclt.converted_gnss.time[i_gps]
    t_imu = imu[i_imu, 0]

    if t_imu < t_gps:
        mm = imu[i_imu, 1:4]
        fm = imu[i_imu, 4:7]
        wm = imu[i_imu, 7:10]
        if pf.get_state_as_numpy()[-1] < 1:
            print 'imu init'
            q = get_orientation_from_magnetic_field(mm, fm)
            pf.initialize_state(q=q, cov_q=np.array([0.1, 0.1, 1]), time=t_imu)
        else:
            print 'prediction'
            pf.predict(fm, np.ones(3)*0.001, wm, t_imu)
            state.append(pf.get_state_as_numpy()[0:3])
        i_imu += 1
        imu_ones = True
    elif imu_ones:
        fix = np.array([nclt.converted_gnss.x[i_gps], nclt.converted_gnss.y[i_gps], nclt.converted_gnss.z[i_gps]])
        if pf.get_state_as_numpy()[-1] < 1:
            print 'gps init'
            pf.initialize_state(p=fix, cov_p=np.ones(3)*20, v=np.zeros(3), cov_v=np.ones(3)*1, ab=np.zeros(3), cov_ab=np.ones(3)*1, g=np.array([0, 0, -9.8]), time=t_gps, initialized=True)
        else:
            print 'correction'
            pf.correct(fix, np.ones(3)*20, t_gps)
        i_gps += 1

    if not imu_ones:
        i_gps += 1

    print 'i_gps:', i_gps
    print 'gps:', np.array([nclt.converted_gnss.x[i_gps], nclt.converted_gnss.y[i_gps], nclt.converted_gnss.z[i_gps]])
    if len(state)>0: print 'est:', state[-1]
    print '------------------------------------------------------------------------\n'

state = np.array(state)
plt.plot(state[:, 0], state[:, 1], label='estimate')
plt.plot(nclt.groundtruth.x, nclt.groundtruth.y, label='groundtruth')
plt.plot(nclt.converted_gnss.x, nclt.converted_gnss.y, label='gps')
plt.legend()
plt.show()