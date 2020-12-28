#!/usr/bin/env python2

import numpy as np
import sympy as sp
import numdifftools as nd
from sympy.utilities import lambdify
from copy import deepcopy

import tf.transformations as tft


class sarray(np.ndarray):
    def __new__(cls, states, *args):
        length = sum([s[1] for s in states])
        obj = np.zeros(length).view(cls)
        obj.states = states
        return obj

    def __init__(self, states=None, val=None):
        i = 0
        for state in self.states:
            self._set_sub_state(i, state[0], state[1])
            i += state[1]
        if val is not None:
            self[:] = np.array(val)

    def _set_sub_state(self, idx, name, size):
        def fun():
            return self[idx:idx + size]
        setattr(self, name, fun)



ns_template = [
    ['p', 2],
    ['v', 2],
    ['o', 1],
]

es_template = [
    ['p', 2],
    ['v', 2],
    ['o', 2]
]

ns = sarray(ns_template, [1,1,2,2,3])
es = sarray(es_template, [10,10,20,20,30,30])

f = [
    lambda s, e, dt: s.p() + s.v() * dt + e.p(),
    lambda s, e, dt: s.v() + e.v(),
    lambda s, e, dt: s.o() + np.linalg.norm(e.o())
]

def fs(s, e, t):
    s = sarray(ns_template, s)
    e = sarray(es_template, e)
    l = [fun(s, e, t).ravel() for fun in f]
    return np.concatenate(l)

def fe(e, s, t):
    s = sarray(ns_template, s)
    e = sarray(es_template, e)
    l = [fun(s, e, t).ravel() for fun in f]
    return np.concatenate(l)

print fs(ns, es, 0.5)
print nd.Jacobian(fs)(ns, es, 0.5)
print nd.Jacobian(fe)(es, ns, 0.5)