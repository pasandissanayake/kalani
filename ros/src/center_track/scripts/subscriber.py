#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
from rospy.numpy_msg import numpy_msg


import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import pickle

bridge = CvBridge()

opt = pickle.load( open( "/home/entc/kalani/ros/src/center_track/scripts/save_5.p", "rb" ) )
opt.load_model = '/home/entc/kalani/ros/src/center_track/models/nuScenes_3Dtracking.pth'

#os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opt.debug = max(opt.debug, 1)
opt.debug = 4
opt.save_video = True
print(opt.task)
#print("opt.debug = ",opt.debug)
#print("opt.skip_first =",opt.skip_first)
detectorr = Detector(opt)

pub = rospy.Publisher('CenterTrack/tracklets', Image, queue_size=10)
print('Done')

def callback(data):
	#rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
	print(rospy.get_time())
	img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
	ret = detectorr.run(img)
	#for keys in ret:
		#print(keys)
	res_img = ret['generic']
	res =[]
	for i in ret['results']:
		#print(i['class'],i['tracking_id'],i['ct'])
		j=i['class'],i['tracking_id'],i['ct']
		res.append(j)
	print(res)
	print(rospy.get_time())
	print('')
	#cv2.imshow('nothing',res_img)
	#cv2.waitKey(500) 
  
# closing all open windows 
	#cv2.destroyAllWindows() 
	image_message = bridge.cv2_to_imgmsg(res_img, "bgr8")
	pub.publish(image_message)


def detector():

	rospy.init_node('detector', anonymous=True)

	rospy.Subscriber('/left_camera_images', numpy_msg(Image), callback)

	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	detector()
