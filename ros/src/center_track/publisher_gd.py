#!/usr/bin/env python
import rospy
from std_msgs.msg import String,Header,Float32MultiArray,MultiArrayDimension

from sensor_msgs.msg import Image,PointCloud2,PointField
from sensor_msgs import point_cloud2
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from rospy.numpy_msg import numpy_msg
#from test_publisher.msg import test_msg

#from kivy.clock import Clock
import time

import pickle

bridge = CvBridge()

image_file_path = "/media/entc/New Volume/DATASETS/Detection-KITTI/data_object_image_2/training/image_2"
velodyne_file_path = "/media/entc/New Volume/DATASETS/Detection-KITTI/data_object_velodyne/training/velodyne"
label_file_path = "/media/entc/New Volume/DATASETS/Detection-KITTI/data_object_label_2/training/label_2"



def GetImageFilesFromDir(dir):
	'''Generates a list of files from the directory'''
	left_files = []
	if os.path.exists(dir):
		for files in os.listdir(dir):
			if files.split(".")[-1] in ['bmp', 'png', 'jpg']:
				#if(files.split(".")[0] == "000025" ):
				left_files.append( os.path.join( dir , files ) )

	left_files.sort()
	return left_files

def GetVeloFilesFromDir(dir):
	'''Generates a list of files from the directory'''
	velodyne_files = []
	if os.path.exists(dir):
		for files in os.listdir(dir):
			if ( files.split(".")[-1] == 'bin'):
				#if(files.split(".")[0] == "000025" ):
				velodyne_files.append( os.path.join( dir , files ) )
	velodyne_files.sort()
	return velodyne_files

def GetLabelFilesFromDir(dir):
	'''Generates a list of files from the directory'''
	label_files = []
	if os.path.exists(dir):
		for files in os.listdir(dir):
			if ( files.split(".")[-1] == 'txt'):
				#if(files.split(".")[0] == "000025" ):				
				label_files.append( os.path.join( dir , files ) )
	label_files.sort()
	return label_files

def talker():

	left_imgs = GetImageFilesFromDir(image_file_path)
	velodynes = GetVeloFilesFromDir(velodyne_file_path)
	labels = GetLabelFilesFromDir( label_file_path  )
	#left_imgs2 = left_imgs[2:7]
	pub = rospy.Publisher('left_camera_images', Image, queue_size=0)
	pub2 = rospy.Publisher('point_clouds', PointCloud2, queue_size=0)
	pub3 = rospy.Publisher('label', String, queue_size=0)
	#pub3 = rospy.Publisher('point_clouds_numpy',Float32MultiArray,queue_size=1)
	rospy.init_node('Perception_Publisher_1', anonymous=True)
	rate = rospy.Rate(1) #10fps
	
	data_velo = []
	#file = open('velo2_pkl', 'rb')
	#velo_data = pickle.load(file)
	#file.close()
	while not rospy.is_shutdown():
		for i in range(len(left_imgs)):
			print("iter",i)

			#read the label
			label_file = open( labels[i] , "r")
			label_data = str( label_file.read() )
			label_msg = String( data = label_data )
			img = cv2.imread(left_imgs[i])
		
			#image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
			image_message = bridge.cv2_to_imgmsg(img, "bgr8")
			ts = rospy.Time.now()
			image_message.header.stamp = ts
			#rospy.loginfo(image_message)

			msg = PointCloud2()
			msg.header.stamp = ts
			msg.header.frame_id = "/map"

			points = np.fromfile(velodynes[i], dtype=np.float32).reshape(-1,4)
			N = len(points)
			msg.height = 1
			msg.width = N

			msg.fields = [
				PointField('x', 0, PointField.FLOAT32, 1),
				PointField('y', 4, PointField.FLOAT32, 1),
				PointField('z', 8, PointField.FLOAT32, 1),
				PointField('i', 12, PointField.FLOAT32, 1)
			]

			msg.is_bigendian = False
			msg.point_step = 16
			msg.row_step = msg.point_step * N
			msg.is_dense = True;
			msg.data = points.tostring()

			"""
			scan = np.fromfile(velodynes[i], dtype=np.float32).reshape(-1,4)
			header = Header()
			header.frame_id = 'my_frame'
			fields = [PointField('x', 0, PointField.FLOAT32, 1),
				  PointField('y', 4, PointField.FLOAT32, 1),
				  PointField('z', 8, PointField.FLOAT32, 1),
				  PointField('i', 12, PointField.FLOAT32, 1)]
			pcl_msg = point_cloud2.create_cloud(header, fields, scan)
			"""

			#data_velo.append(pcl_msg)

    		#scoresmsg = Float32MultiArray()
			#scoresmsg.data = scan
			#scoresmsg.stamp = ts
			#pcl_msg = velo_data[i]
			#pcl_msg.header.stamp = ts
			pub.publish(image_message)
			pub2.publish( msg )
			pub3.publish( label_msg )
			#pub3.publish(scoresmsg)
			print(rospy.get_time())
			rate.sleep()

			print("Messages published ")
			print("\n")
    	print('//////////////////////////////////////////////////////////////////////////////////////////////////////done////////////')
    	'''file = open('velo2_pkl', 'wb')
		pickle.dump(data_velo, file)
		file.close()
		print('..........................................Done..............................................')
		break'''
			

		

if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
