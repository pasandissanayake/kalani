import numpy as np
from utilities import *
from kaist_datahandle import *
import time

start = time.time()
kd = KAISTData()
print 'loading started'
kd.load_data(groundtruth=True)
pl = kd.get_player()
print pl.next()
print 'loading done. time elapsed: {} s'.format(time.time() - start)
