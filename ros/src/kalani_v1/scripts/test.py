from datasetutils.kaist_data_conversions import KAISTDataConversions, KAISTData
import matplotlib.pyplot as plt
import numpy as np


kd = KAISTData('/home/entc/kalani/data/kaist/urban17')
plt.plot(kd.groundtruth.x,kd.groundtruth.y)
plt.plot(kd.converted_gnss.x,kd.converted_gnss.y)
# plt.plot(range(100), 2*np.array(range(100)))
plt.show()

