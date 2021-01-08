#!/usr/bin/env python2

import numpy as np
import sympy as sp
import numdifftools as nd
from sympy.utilities import lambdify
from copy import deepcopy
from utilities import *

import matplotlib.pyplot as plt

import tf.transformations as tft
from kalman_filter import KalmanFilter

POINTSpS = 1 # points per second
DURATION = 1000  # duration in seconds
POINTS = POINTSpS * DURATION
t = np.linspace(0.1, DURATION, POINTS)
# g = np.array([np.cos(0.2 * t) + 0.5 * np.cos(0.5 * t), t]).T
g = np.array([np.cos(0.2 * t) + 0.5 * np.cos(0.5 * t), np.sin(0.2 * t)+3]).T
g = g * 4

v_std = 1e-0
v = np.zeros((len(g), 2))
v[0, 0] = 0
v[1:,0] = np.diff(g[:,0])/(t[1]-t[0])
v[0, 1] = 0
v[1:,1] = np.diff(g[:,1])/(t[1]-t[0])
v_noise = np.random.normal(scale=v_std, size=(POINTS, 2))
v[:,0] = v[:,0] + v_noise[:,0]
v[:,1] = v[:,1] + v_noise[:,1]

g_std = 10
g_noise = np.random.normal(scale=g_std, size=(POINTS, 2))
noisy_g = np.array([g[:,0] + g_noise[:, 0], g[:, 1] + g_noise[:, 1]]).T

b_std = 1e-4

d_std = 1e-0
g_dif = np.diff(g, axis=0)
g_dif = np.concatenate([[[0, 0]], g_dif])
g_dif_noise = np.random.normal(scale=d_std, size=(POINTS, 2))
g_dif[:,0] = g_dif[:,0] + g_dif_noise[:,0]
g_dif[:,1] = g_dif[:,1] + g_dif_noise[:,1]

def motion_model(ts, mmi, pn, dt):
    return np.array([
        ts.p() + (mmi.v() + pn.v()) * dt,
        # ts.p() + (mmi.v() + pn.v()) * dt + ts.b(),
        # ts.b() + pn.b()
    ]).ravel()

def combination(ns, es):
    return np.array([
        ns.p() + es.p(),
        # ns.b() + es.b()
    ]).ravel()

def difference(ns1, ns0):
    return np.array([
        ns1.p() - ns0.p(),
        # ns1.b() - ns0.b()
    ]).ravel()
kf = KalmanFilter(
    [
        ['p', 2],
        # ['b', 2]
    ],

    [
        ['p', 2],
        # ['b', 2]
    ],

    [
        ['v', 2],
        # ['b', 2]
    ],

    [
        ['v', 2]
    ],

    motion_model,
    combination,
    difference
)
kf.initialize(
    [
        ['p', [-2,0], np.eye(2) * g_std**2, 0.0],
        # ['b', [0,0], np.eye(2) * b_std**2, 0,0]
    ]
)

out = []
out_cov = []
out.append(kf.get_current_state())
out_cov.append(kf.get_current_cov())
prediction_time = []
correction_time = []
for i in range(len(g)):
    # instr = raw_input('press enter')
    sw = Stopwatch()
    sw.start()
    kf.predict(
        v[i],
        np.diag(np.concatenate([
            np.ones(2) * 1 * v_std**2,
            # np.ones(2) * b_std**2
        ])),
        t[i],
        input_name='velocity'
    )
    prediction_time.append(sw.stop())

    sw = Stopwatch()
    sw.start()

    # def meas_fun(ns):
    #     return ns.p()
    # if i > 2:
    #     kf.correct_absolute(
    #         meas_fun,
    #         noisy_g[i-1],
    #         np.eye(2) * 1 * g_std**2,
    #         t[i-1],
    #         measurement_name='position'
    #     )

    def meas_rel(ns1, ns0):
        return ns1.p() - ns0.p()
    if i > 2:
        kf.correct_relative(
            meas_rel,
            g_dif[i],
            np.eye(2) * d_std**2,
            t[i],
            t[i-1],
            measurement_name='position'
        )
    correction_time.append(sw.stop())
    out.append(kf.get_current_state())
    out_cov.append(kf.get_current_cov())

out = np.array(out)
out_cov = np.array(out_cov)
t = np.concatenate([[0], t])

print 'rms error in x: {}m,  rms error in y: {}m'.format(np.sqrt(np.mean((out[1:,0]-g[:,0])**2)), np.sqrt(np.mean((out[1:,1]-g[:,1])**2)))
print 'prediction time: {}s,  correction time: {}s'.format(np.average(prediction_time), np.average(correction_time))

fig1, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(t[1:], g[:, 0], label='x ground truth')
ax1.plot(t[1:], noisy_g[:, 0], '.', label='x measurement')
ax1.plot(t, out[:, 0], label='x estimate')
ax1.plot(t, out[:, 0] + 3 * np.sqrt(out_cov[:, 0]), '--', label='x upper bound')
ax1.plot(t, out[:, 0] - 3 * np.sqrt(out_cov[:, 0]), '--', label='x lower bound')
ax1.legend()
ax1.grid()

ax2.plot(t[1:], g[:, 1], label='y ground truth')
ax2.plot(t[1:], noisy_g[:, 1], '.', label='y measurement')
ax2.plot(t, out[:, 1], label='y estimate')
ax2.plot(t, out[:, 1] + 3 * np.sqrt(out_cov[:, 3]), '--', label='y upper bound')
ax2.plot(t, out[:, 1] - 3 * np.sqrt(out_cov[:, 3]), '--', label='y lower bound')
ax2.legend()
ax2.grid()

fig2, (bx1, bx2) = plt.subplots(2, 1)

bx1.plot(t[1:], v[:, 0], '.', label='x measured velocity')
bx1.plot(t[1:], v[:, 0]-v_noise[:, 0], label='x actual velocity')
bx1.legend()
bx1.grid()

bx2.plot(t[1:], v[:, 1], '.', label='y measured velocity')
bx2.plot(t[1:], v[:, 1]-v_noise[:, 1], label='y actual velocity')
bx2.legend()
bx2.grid()

# fig3, (cx1, cx2) = plt.subplots(2, 1)
#
# cx1.plot(t, out[:, 2], label='bias x')
# cx1.legend()
# cx1.grid()
#
# cx2.plot(t, out[:, 3], label='bias y')
# cx2.legend()
# cx2.grid()

plt.show()