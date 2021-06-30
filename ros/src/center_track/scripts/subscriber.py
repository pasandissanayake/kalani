#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')

import threading
from numpy.core.numeric import indices
import time
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String,Header,Float32MultiArray, MultiArrayDimension
from extraction_utils import Box3D , read_calib_file , map_box_to_image , draw_projected_box3d , draw_lidar
from extraction_utils  import load_label , load_image , draw_boxes
from object_extraction import lidar_extraction
import _init_paths
import mayavi.mlab as mlab
import pickle
import ros_np_multiarray as rnm
from label_utils import load_cam2cam , load_velo2cam , velo_2_img_projection
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

pub_img = rospy.Publisher('CenterTrack/tracklets', Image, queue_size=1)
pub_lidar = rospy.Publisher('CenterTrack/point_cloud', PointCloud2, queue_size=1)
print('Done')

v2c_filepath = "/media/entc/New Volume/data/kitti_tracklets/2011_09_26/calib_velo_to_cam.txt"
c2c_filepath = "/media/entc/New Volume/data/kitti_tracklets/2011_09_26/calib_cam_to_cam.txt"
#define parameters

img_height = 375
img_width = 1242
objects , lines = [] , [] 

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
	center = [img.shape[0] / 2   , img.shape[1]/2 ]
	x = center[1] 
	y = center[0] 
	print(img.shape , x, y)
	#crop_img = img[int(y - img_height/2. ):int(y+ img_height/2.) , int(x - img_width/2.):int(x+img_width/2.)]
	crop_img = img
	#img = cv2.resize(img, (img_width , img_height) , interpolation = cv2.INTER_AREA)
	print(img_height , img_width , crop_img.shape)    # 375 1242  --> 512 1392
	start_time = time.time()

	ret = detectorr.run(crop_img)

	end_time = time.time()

	print("Total Image Execution Time : ",end_time - start_time)

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

	###########################################################################################################
	"""
	for i_line in lines :
		data = i_line.split(' ')
		data[1:] = [float(x) for x in data[1:]]
		ct_data = [ 0 ] * 12

		ct_data[11] = data[0]

        # extract 2d bounding box in 0-based coordinates
		ct_data[0] = data[4]  # left
		ct_data[1] = data[5]  # top
		ct_data[2] = data[6]  # right
		ct_data[3] = data[7]  # bottom

		# extract 3d bounding box information
		ct_data[4] = data[8]  # box height
		ct_data[5] = data[9]  # box width
		ct_data[6] = data[10]   # box length (in meters)
		ct_data[7] , ct_data[8] , ct_data[9] = data[11]  , data[12]  , data[13]  # location (x,y,z) in camera coord.
		ct_data[10] = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
		print("target",ct_data)
		if( ct_data[9]  < 100 ):

			objects.append( Box3D( ct_data ) )

	objects =[]

	###########################################################################################################
	
	for i_idx , i in enumerate( ret['results']):
		
		ct_data = [ 0 ] * 12
		ct_data[0],ct_data[1],ct_data[2],ct_data[3] = i['bbox'][0],i['bbox'][1] ,i['bbox'][2],i['bbox'][3]
		ct_data[4] , ct_data[5] , ct_data[6] = i['dim'][0] +1.5 , i['dim'][1]+2.5 , i['dim'][2] +1.5
		ct_data[7] , ct_data[8] , ct_data[9] = i['loc'][0] - 0.15  , i['loc'][1]+0.55 , i['loc'][2] - 7.8  # 1 (0.65) + 0.45 + 1.75
		ct_data[10] = i['rot_y']
		ct_data[11] = i['class']
		print("predict",ct_data)
		#and ( i_idx <10 ) 
		if( ( i['loc'][2] < 100) ):

			objects.append( Box3D(ct_data) )

	print("Number of objects " , len(objects))
	
	if( len(objects)!=0 ):

		# get the index corresponds to the point cloud
		lidar_cloud , index_array = lidar_extraction(  lidar_np , calib , img_width, img_height , objects )
		cloud_msg = xyzi_array_to_pointcloud2(  lidar_cloud , point_array["i"][ index_array ] , rospy.Time.from_sec(time_sec) , frame_id )
		#cloud_msg = xyzi_array_to_pointcloud2(  lidar_cloud , None , rospy.Time.from_sec(time_sec) , frame_id )
	else :
		cloud_msg =  array_to_pointcloud2( point_array , rospy.Time.from_sec(time_sec) , frame_id  )

	pub_lidar.publish( cloud_msg )
	"""
	#out_img = render_image_with_boxes( crop_img , objects , calib)

	print(rospy.get_time())
	print('')

	image_message = bridge.cv2_to_imgmsg( res_img , "bgr8")
	pub_img.publish(image_message)

##############################################################################################################

def pointcloud_callback( pointcloud2 ):

	global time_sec , frame_id  , point_array , lidar_np 

	time_sec = pointcloud2.header.stamp.to_sec()
	frame_id = 'lidar'
	#pointcloud2.header.frame_id = frame_id
	
	start_time = time.time()
	point_array = pointcloud2_to_array( pointcloud2 )
	lidar_np = get_xyz_points( point_array )

	end_time = time.time()

	print("Total Lidar numpy Execution Time : ",end_time - start_time)
	
	if( len(tracklet2d)!=0 ):

		# get the index corresponds to the point cloud
		lidar_cloud , index_array = lidar_extraction(  lidar_np , calib , img_width, img_height , tracklet2d , depth2d )
		#cloud_msg = xyzi_array_to_pointcloud2(  lidar_cloud , point_array["i"][ index_array ] , rospy.Time.from_sec(time_sec) , frame_id )
		cloud_msg = xyzi_array_to_pointcloud2(  lidar_cloud , None , rospy.Time.from_sec(time_sec) , frame_id )
	else :
		cloud_msg =  array_to_pointcloud2( point_array , rospy.Time.from_sec(time_sec) , frame_id  )

	pub_lidar.publish( cloud_msg )	
	
##########################################################################################################

def label_callback( String ) :

	global lines ;
	lines = str(String.data).split("\n")[:-1]
	
	# load as list of Object3D

def tracklet_callback( tracklet ):
	print("Tracklet Data")
	data = rnm.to_numpy_f32(  tracklet )

	global tracklet2d , depth2d 

	tracklet2d = []
	depth2d = []
	for i in data :
		point = i.T
		#chk,_ , depth = check._Kitti_util__velo_2_img_projection(point)
		chk  , depth = velo_2_img_projection(point  , v2c_file , c2c_file )
		chk = np.concatenate((chk[0].reshape(1,-1) , chk[1].reshape(1,-1)  ) , axis=0).T
		tracklet2d.append(chk)
		depth2d.append( depth )

	print( tracklet2d )
	

def detector():

	rospy.init_node('detector', anonymous=True)

	#while not rospy.is_shutdown():

	rospy.Subscriber('/camera/colour/image_raw', numpy_msg(Image), callback) #/left_camera_images  /camera/colour/image_raw
	rospy.Subscriber('/os1_points' , PointCloud2 , pointcloud_callback ) #/point_clouds   os1_points
	rospy.Subscriber("/label" , String , label_callback )
	rospy.Subscriber("/tracklets",  Float32MultiArray  , tracklet_callback )
	# spin() simply keeps python from exiting until this node is stopped
	#r = rospy.Rate(1)
	#r.sleep()
	rospy.spin()

###########################################################################################################

if __name__ == '__main__':

	# Load calibration
	calib = read_calib_file('scripts/data/000114_calib.txt')

	v2c_file = load_velo2cam( v2c_filepath )
	c2c_file = load_cam2cam( c2c_filepath )

	#run the detector
	detector()
