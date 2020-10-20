import numpy as np
import matplotlib.pyplot as plt

import tf.transformations as tft

from datasetutils.nclt_data_conversions import *

class Particle:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0,0,0,1])

        self.g = np.array((0,0,-9.8))

        self.P = np.eye(6)

        self.weight = 0
        self.time = -1
        self.initialized = [False * 3]

    def initialize(self, time, p=None, cov_p=None, v=None, cov_v=None, q=None, weight=None):
        if p is not None:
            self.p = np.array(p)
            self.P[0:3, 0:3] = np.diag(cov_p)
            self.initialized[0] = True

        if v is not None:
            self.v = np.array(v)
            self.P[3:6, 3:6] = np.diag(cov_v)
            self.initialized[1] = True

        if q is not None:
            self.q = np.array(q)
            self.weight = weight
            self.initialized[2] = True

        self.time = time

    def predict(self, time, am, cov_am):
        dt = time - self.time
        R = tft.quaternion_matrix(self.q)[0:3, 0:3]

        self.p = self.p + self.v * dt + 0.5 * (np.matmul(R, am) + self.g) * dt**2
        self.v = self.v + (np.matmul(R, am) + self.g) * dt

        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        B = np.concatenate([0.5*R*dt**2, R*dt], axis=0)
        self.P = np.matmul(F, np.matmul(self.P, F.T)) + np.matmul(B, np.matmul(np.diag(cov_am), B.T))

