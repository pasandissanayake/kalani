#!/usr/bin/env python

import rospy
import tf.transformations as tft
import tf
import numpy as np
from scipy.interpolate import interp1d
from datasetutils.kaist_data_conversions import KAISTDataConversions, KAISTData
import matplotlib.pyplot as plt

ds = KAISTData('/home/pasan/kalani/data/kaist/urban-17', gt=True, gnss=True)
plt.plot(ds.groundtruth.x, ds.groundtruth.y)
plt.show()
