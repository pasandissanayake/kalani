import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from kaist_datahandle import *
import tf.transformations as tft


e1 = np.array([np.pi/3, 0, np.pi/2])
e2 = np.array([np.pi/4, 0, np.pi/4])

q1 = tft.quaternion_about_axis(tft.vector_norm(e1), tft.unit_vector(e1))
q2 = tft.quaternion_about_axis(tft.vector_norm(e2), tft.unit_vector(e2))

de = e1+e2
qd = tft.quaternion_multiply(q2, q1)

print 'de: {}'.format(np.array(tft.euler_from_quaternion(tft.quaternion_about_axis(tft.vector_norm(de), tft.unit_vector(de)))))
print 'qe: {}'.format(np.array(tft.euler_from_quaternion(qd)))