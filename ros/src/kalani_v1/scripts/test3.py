
import numpy as np
import autograd.numpy as anp
from autograd import jacobian




def f(x):
    qw, qx, qy, qz = x[6:10]

    r = anp.array([
        [qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2]
    ]).T
    return anp.matmul(r, x[16:19] - x[0:3])


jacob = jacobian(f)
state = np.concatenate([np.ones(6)*2, [1, 0, 0, 0], np.ones(12) * 5, [1, 0, 0, 0], np.zeros(6)])
print np.array(jacob(state))