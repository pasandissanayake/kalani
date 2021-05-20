import numpy as np
from utilities import *
import tf.transformations as tft

# qa = np.array((1,0,0,0))
# qb = np.array((0,1,0,0))


a = np.array([
    [ 4,  0,  3],
    [20, 30, 10],
    [ 2,  0,  1]
])

b = np.array([6, 8, 5])

print np.dot(a, b)


qa = np.random.rand(4)
qb = np.random.rand(4)

qa = qa / np.linalg.norm(qa)
qb = qb / np.linalg.norm(qb)

qt = tft.quaternion_multiply(qa, qb)
qm = np.dot(quaternion_left_multmat(qa), qb)
qn = np.dot(quaternion_right_multmat(qb), qa)

# print qa
# print qb
# print qt
# print qm, np.allclose(qt, qm)
# print qn, np.allclose(qt, qn)

print jacobian_of_axisangle_wrt_q((0,0,0,1))