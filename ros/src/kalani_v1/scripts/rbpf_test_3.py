import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import copy
from copy import deepcopy
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
    nPoints = np.ndim(points)
    nMeans = np.ndim(means)
    nCovariances = np.ndim(covariances)
    if nPoints > 1 and nMeans > 1 and nCovariances > 2:
        print 'many points, many means'
        results = []
        for i in range(len(points)):
            results.append(multivariate_normal.pdf(points[i], mean=means[i], cov=covariances[i]))
        return np.array(results)
    elif nPoints > 1 and nMeans < 2 and nCovariances < 3:
        print 'many points, one mean'
        results = []
        for i in range(len(points)):
            results.append(multivariate_normal.pdf(points[i], mean=means, cov=covariances))
        return np.array(results)
    elif nPoints < 2 and nMeans > 1 and nCovariances > 2:
        print 'one point, many means'
        results = []
        for i in range(len(means)):
            results.append(multivariate_normal.pdf(points, mean=means[i], cov=covariances[i]))
        return np.array(results)
    else:
        print 'one point, one mean'
        return multivariate_normal.pdf(points, mean=means, cov=covariances)


class RB_Particle_Filter_V1():
    def __init__(self):
        self.NO_OF_PARTICLES = 5 * 3

        self._pf_state = np.zeros([self.NO_OF_PARTICLES, 4])
        self._pf_weights = np.ones(self.NO_OF_PARTICLES) / self.NO_OF_PARTICLES
        self._kf_state = np.zeros([self.NO_OF_PARTICLES, 9])
        self._kf_covariance = np.zeros([self.NO_OF_PARTICLES, 9, 9])
        self._filter_time = -1
        self._mean_state = np.zeros(13)
        self._mean_covariance = np.zeros((9, 9))
        self._mean_filter_time = -1
        self._filter_initialized = False
        self._g = np.array([0, 0, -9.8])

        self._a, self._b = np.mgrid[-0.1:0.1:0.001, -0.1:0.1:0.001]
        self._pos = np.dstack((self._a, self._b))
        self._winners = np.zeros(3)


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
        self._pf_state = quaternion_multiply(dq, self._pf_state)

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
        if time - self._filter_time < -1:
            print 'old correction!'
            return

        H = np.zeros((2,9))
        H[0:2, 0:2] = np.eye(2)

        old_kf_state = deepcopy(self._kf_state)

        y = fix - matrix_vector_multiply(H, self._kf_state)
        S = matrix_matrix_multiply(H, matrix_matrix_multiply(self._kf_covariance, H.T)) + var_fix
        K = matrix_matrix_multiply(self._kf_covariance, matrix_matrix_multiply(H.T, matrix_inverse(S)))
        self._kf_covariance = matrix_matrix_multiply(np.eye(9) - matrix_matrix_multiply(K, H), self._kf_covariance)
        self._kf_covariance = 0.5 * (self._kf_covariance + matrix_transpose(self._kf_covariance))
        self._kf_state = self._kf_state + matrix_vector_multiply(K, y)

        pdf = multivariate_normal_pdf(fix - matrix_vector_multiply(H, self._kf_state), np.zeros(2), S[0])
        print 'S:\n', S[0]

        rv = multivariate_normal([0, 0], S[0])
        plt.contour(self._a + fix[0], self._b + fix[1], rv.pdf(self._pos))

        print 'updating weights...'
        self._pf_weights = self._pf_weights * pdf
        self._pf_weights = self._pf_weights / np.sum(self._pf_weights)

        max_proba_i = np.argmax(pdf)
        min_error_i = np.argmin(np.linalg.norm(y, axis=1))
        print 'proba_max:', y[max_proba_i], 'norm:', np.linalg.norm(y[max_proba_i]), 'pdf:', pdf[max_proba_i], 'new wgt:', self._pf_weights[max_proba_i]
        print 'error_min:', y[min_error_i], 'norm:', np.linalg.norm(y[min_error_i]), 'pdf:', pdf[min_error_i], 'new wgt:', self._pf_weights[min_error_i]

        # if np.max(self._pf_weights) < 1e-3:
        if 0 < 1:
            print 'resampling...'
            nsamples = np.round(self._pf_weights * self.NO_OF_PARTICLES / np.sum(self._pf_weights))
            print 'normalized weights range:', np.min(self._pf_weights / np.sum(self._pf_weights)), np.max(self._pf_weights / np.sum(self._pf_weights))
            print 'nsamples range:', np.min(nsamples), np.max(nsamples)

            if np.sum(nsamples) != self.NO_OF_PARTICLES:
                deficit = self.NO_OF_PARTICLES - np.sum(nsamples)
                i = np.argmax(nsamples)
                nsamples[i] = nsamples[i] + deficit

            new_pf_state = []
            new_kf_state = []
            new_kf_covariance = []
            self._winners = []

            # re-sample proportionate to weight
            # for i in range(len(nsamples)):
            #     for j in range(int(nsamples[i])):
            #         new_pf_state.append(deepcopy(self._pf_state[i]))
            #         new_kf_state.append(deepcopy(self._kf_state[i]))
            #         new_kf_covariance.append(deepcopy(self._kf_covariance[i]))
            #         self._winners.append(deepcopy(old_kf_state[i, 0:3]))

            # re-sample by maximum weight
            idx = np.argmax(nsamples)
            for i in range(self.NO_OF_PARTICLES):
                new_pf_state.append(self._pf_state[idx])
                new_kf_state.append(self._kf_state[idx])
                new_kf_covariance.append(self._kf_covariance[idx])
                self._winners.append(deepcopy(old_kf_state[idx, 0:3]))

            self._pf_state = np.array(new_pf_state)
            self._winners = np.array(self._winners)

            euler = euler_from_quaternion(self._pf_state)
            noise = np.random.multivariate_normal(np.zeros(3), np.diag([0.001, 0.001, 0.01]), self.NO_OF_PARTICLES)
            neulers = noise + euler
            self._pf_state = quaternion_from_euler(neulers[:, 0], neulers[:, 1], neulers[:, 2])

            self._pf_weights = np.ones(self.NO_OF_PARTICLES) / self.NO_OF_PARTICLES
            self._kf_state = np.array(new_kf_state)
            self._kf_covariance = np.array(new_kf_covariance)

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
        self._mean_covariance = np.array(p)
        self._mean_filter_time = copy.deepcopy(self._filter_time)


    def get_state_as_numpy(self):
        return np.concatenate([self._mean_state[0:6], self._mean_state[9:13], self._mean_state[6:9], [0,0,0, self._mean_filter_time, float(self._filter_initialized)]])

    def get_positions_of_all_particles_as_numpy(self):
        return np.array(self._kf_state[:, 0:3])

    def get_winners(self):
        return deepcopy(self._winners)



pf = RB_Particle_Filter_V1()

nclt = NCLTData('/home/pasan/kalani/data/nclt/2013-01-10')

imur = np.loadtxt('/home/pasan/kalani/data/nclt/2013-01-10/ms25.csv', delimiter=',')
imu = np.zeros((len(imur), 10))
imu[:, 0] = imur[:, 0] * 1e-6
for j in range(3):
    i = j * 3
    imu[:, 1 + i] = imur[:, 2 + i]
    imu[:, 2 + i] = imur[:, 1 + i]
    imu[:, 3 + i] = -imur[:, 3 + i]

gt_lim = 1000
gps_lim = 4
# plt.plot(nclt.groundtruth.x[0:gt_lim], nclt.groundtruth.y[0:gt_lim], label='groundtruth')
plt.plot(nclt.converted_gnss.x[0:gps_lim], nclt.converted_gnss.y[0:gps_lim], label='gps')
plt.legend()
plt.ion()
plt.show()

i_gps = 0
i_imu = 0
prev_time = -1
imu_ones = False
for i in range(1000):
    skip_pause = False
    t_gps = nclt.converted_gnss.time[i_gps]
    t_imu = imu[i_imu, 0]

    if t_imu < t_gps:
        if prev_time > t_imu:
            i_imu += 1
            continue
        prev_time = t_imu

        mm = imu[i_imu, 1:4]
        fm = imu[i_imu, 4:7]
        wm = imu[i_imu, 7:10]
        if pf.get_state_as_numpy()[-1] < 1:
            print 'imu init'
            q = get_orientation_from_magnetic_field(mm, fm)
            pf.initialize_state(q=q, cov_q=np.array([0.01, 0.01, 0.1]), time=t_imu)
            skip_pause = True
        else:
            print 'prediction'
            pf.predict(fm, np.ones(3)*0.001, wm, t_imu)

            state = pf.get_state_as_numpy()[0:3]
            particles = pf.get_positions_of_all_particles_as_numpy()
            plt.plot(state[0], state[1], linestyle='dashed', marker='o', color='black')
            plt.plot(particles[:, 0], particles[:, 1], '.')
            plt.xlabel('step: ' + str(i) + ' prediction')
            plt.draw()

        i_imu += 1
        imu_ones = True
    elif imu_ones:
        if prev_time > t_gps:
            i_gps += 1
            continue
        prev_time = t_gps

        fix = np.array([nclt.converted_gnss.x[i_gps], nclt.converted_gnss.y[i_gps], nclt.converted_gnss.z[i_gps]])
        if pf.get_state_as_numpy()[-1] < 1:
            print 'gps init'
            pf.initialize_state(p=fix, cov_p=np.ones(3)*200, v=np.zeros(3), cov_v=np.ones(3)*1, ab=np.zeros(3), cov_ab=np.ones(3)*0, g=np.array([0, 0, -9.8]), time=t_gps, initialized=True)
            skip_pause = True
        else:
            print 'correction'
            particles = []
            particles.append(pf.get_positions_of_all_particles_as_numpy())
            pf.correct(fix[0:2], np.ones(2)*200, t_gps)

            particles.append(pf.get_positions_of_all_particles_as_numpy())
            particles = np.array(particles)
            for j in range(pf.NO_OF_PARTICLES):
                plt.plot([particles[0, j, 0], particles[1, j, 0]], [particles[0, j, 1], particles[1, j, 1]], linestyle='dashed', marker='^', color='blue')

            state = pf.get_state_as_numpy()[0:3]
            plt.plot(state[0], state[1], linestyle='dashed', marker='o', color='black')

            winners = pf.get_winners()
            plt.scatter(winners[:, 0], winners[:, 1], marker='o', color='red', s=80, facecolors='none')

            plt.plot(fix[0], fix[1], 's')
            plt.xlabel('step: ' + str(i) + ' correction')
            plt.draw()

        i_gps += 1

    if not imu_ones:
        i_gps += 1
    # print 'i_gps:', i_gps
    # print 'gps:', np.array([nclt.converted_gnss.x[i_gps], nclt.converted_gnss.y[i_gps], nclt.converted_gnss.z[i_gps]])
    # if len(state)>0: print 'est:', state[-1]
    print '------------------------------------------------------------------------\n'
    if not skip_pause: pause = raw_input('press enter to continue...')
print 'end of loop'
a = raw_input()