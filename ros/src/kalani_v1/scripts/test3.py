#!/usr/bin/env python2

import numpy as np
import tf.transformations as tft

p = tft.quaternion_from_euler(0, 0, np.pi/4)
q = tft.quaternion_from_euler(0, 0, np.pi/2)
r = tft.quaternion_multiply(q, tft.quaternion_conjugate(p))

r_eu = tft.euler_from_quaternion(r)
print np.array(r_eu) * 180 / np.pi