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
from sensor_msgs.msg import PointCloud2, PointField
from extraction_utils import Box3D , read_calib_file , map_box_to_image , draw_projected_box3d , draw_lidar
from extraction_utils  import load_label , load_image , draw_boxes
from object_extraction import lidar_extraction
import _init_paths
import mayavi.mlab as mlab
import pickle

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import pickle
from point_cloud_utils import *

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

pub_img = rospy.Publisher('CenterTrack/tracklets', Image, queue_size=10)
pub_lidar = rospy.Publisher('point_clouds', PointCloud2, queue_size=10)
print('Done')


#define parameters

img_height = 375
img_width = 1242
objects = [] 

#############################################################################################################

def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 8 :
            continue
        box3d_pixelcoord , _ = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    return img1

###########################################################################################################

def callback(data):

	#rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
	print(rospy.get_time())
	img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
	#img_resize = img[ :img_height , :img_width , : ]
	#img = cv2.resize(img, (img_width , img_height) , interpolation = cv2.INTER_AREA)
	print(img_height , img_width , img.shape)    # 375 1242  --> 512 1392
	ret = detectorr.run(img)

	"""
	Keys (['score', 'class', 'ct', 'tracking', 'bbox', 'dep', 
	'dim', 'alpha', 'loc', 'rot_y', 'tracking_id', 'age', 'active'])

	Boundig box 
	[1131.2794   136.80562 1172.6478   208.14354]
	3D dimesnion 
	[1.7388322 0.663208  0.7341847]
	3D location 
	[19.115303    0.45803723 22.67984   ]
	rotation y 
	1.485504475882495
	"""

	res_img = ret['generic']
	global objects ;

	objects =[]

	for i in ret['results']:
		
		ct_data = [ 0 ] * 12
		ct_data[0],ct_data[1],ct_data[2],ct_data[3] = i['bbox'][0],i['bbox'][1],i['bbox'][2],i['bbox'][3]
		ct_data[4] , ct_data[5] , ct_data[6] = i['dim'][0] , i['dim'][1] , i['dim'][2]
		ct_data[7] , ct_data[8] , ct_data[9] = i['loc'][0] , i['loc'][1] , i['loc'][2]
		ct_data[10] = i['rot_y']
		ct_data[11] = i['class']
		if( i['loc'][2] < 80 ):

			objects.append( Box3D(ct_data) )

	out_img = render_image_with_boxes( img , objects , calib)

	print(rospy.get_time())
	print('')

	image_message = bridge.cv2_to_imgmsg( out_img , "bgr8")
	pub_img.publish(image_message)

##############################################################################################################

def pointcloud_callback( pointcloud2 ):

	point_array = pointcloud2_to_array( pointcloud2 )
	lidar_np = get_xyz_points( point_array )
	print( point_array.dtype.names )
	if( len(objects)!=0 ):

		# get the index corresponds to the point cloud
		lidar_cloud = lidar_extraction(  lidar_np , calib , img_width, img_height , objects )
		#point_array["x"] , point_array["y"] , point_array["z"] = point_array["x"][ lidar_cloud ] , point_array["y"][ lidar_cloud ] , point_array["z"][ lidar_cloud ] 
		print(len( lidar_cloud ))
		#convert cloud array to pointcloud message
		#pt_cloud = array_to_pointcloud2(  point_array )

		with open('test.pkl','wb') as f:
			pickle.dump( lidar_cloud , f)
	#else :
		#pt_cloud =  array_to_pointcloud2( point_array )

	#pub_lidar.publish( pt_cloud )	

##########################################################################################################

def detector():

	rospy.init_node('detector', anonymous=True)

	rospy.Subscriber('/left_camera_images', numpy_msg(Image), callback) #/left_camera_images  /camera/right/image_raw
	rospy.Subscriber( "/point_clouds" , PointCloud2 , pointcloud_callback )

	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

###########################################################################################################

if __name__ == '__main__':

	# Load calibration
	calib = read_calib_file('data/000114_calib.txt')
	#run the detector
	detector()
