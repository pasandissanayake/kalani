#!/usr/bin/env python2

import numpy as np
from datasetutils.nclt_data_conversions import *
from constants import Constants

class Foo:
    def __init__(self, states):
        for state in states:
            self.__dict__[state[0]] = np.zeros(state[1])

states = [
    ['a', 3],
    ['b', 4]
]
foo = Foo(states)
foo.a = np.array((2,1,3))
print foo.a, foo.b

ds = NCLTData(Constants.NCLT_DATASET_DIRECTORY)
gt = ds.groundtruth
print gt.x[0], gt.y[0], gt.z[0]

np.savetxt('apple', np.array([0,0]), delimiter=',')
with open("apple", "ab") as f:
    np.savetxt(f, np.array([gt.x[0:5], gt.y[0:5]]), delimiter=',')