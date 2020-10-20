#!/usr/bin/env python2

import numpy as np

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