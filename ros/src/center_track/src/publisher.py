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

image_file_path = "/home/entc/kalani-data/kitti/Kitti/object/training"
velodyne_file_path = "/home/entc/kalani-data/kitti/Kitti/object/training"

def GetImageFilesFromDir(dir):
	'''Generates a list of files from the directory'''
	left_files = []
	if os.path.exists(dir):
		for path, names, files in os.walk(dir):
			for f in files:
				if os.path.splitext(f)[1] in ['.bmp', '.png', '.jpg']:
					left_files.append( os.path.join( path, f ) )
	left_files.sort()
	return left_files

def GetVeloFilesFromDir(dir):
	'''Generates a list of files from the directory'''
	velodyne_files = []
	if os.path.exists(dir):
		for path, names, files in os.walk(dir):
			for f in files:
				if (os.path.splitext(f)[1] == '.bin'):
					velodyne_files.append( os.path.join( path, f ) )
	velodyne_files.sort()
	return velodyne_files

def talker():
	left_imgs = GetImageFilesFromDir(image_file_path)
	velodynes = GetVeloFilesFromDir(velodyne_file_path)
	#left_imgs2 = left_imgs[2:7]
	pub = rospy.Publisher('left_camera_images', Image, queue_size=1)
	pub2 = rospy.Publisher('point_clouds', PointCloud2, queue_size=1)
	#pub3 = rospy.Publisher('point_clouds_numpy',Float32MultiArray,queue_size=1)
	rospy.init_node('Perception_Publisher', anonymous=True)
	rate = rospy.Rate(2) #10fps
	
	data_velo = []
	#file = open('velo2_pkl', 'rb')
	#velo_data = pickle.load(file)
	#file.close()
	while not rospy.is_shutdown():
		for i in range(len(left_imgs)):
			print(rospy.get_time())
			img = cv2.imread(left_imgs[i])
			#image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
			image_message = bridge.cv2_to_imgmsg(img, "bgr8")
			ts = rospy.Time.now()
			image_message.header.stamp = ts
			print(image_message.header.stamp)
			#rospy.loginfo(image_message)
			scan = (np.fromfile(velodynes[i], dtype=np.float32)).reshape(-1,4)
			header = Header()
			header.frame_id = 'my_frame'
			fields = [PointField('x', 0, PointField.FLOAT32, 1),
				  PointField('y', 4, PointField.FLOAT32, 1),
				  PointField('z', 8, PointField.FLOAT32, 1),
				  PointField('i', 12, PointField.FLOAT32, 1)]
			pcl_msg = point_cloud2.create_cloud(header, fields, scan)

			data_velo.append(pcl_msg)
			#scoresmsg = Float32MultiArray()
			#scoresmsg.data = scan
			#scoresmsg.stamp = ts
			#pcl_msg = velo_data[i]
			pcl_msg.header.stamp = ts
			pub.publish(image_message)
			pub2.publish(pcl_msg)
			#pub3.publish(scoresmsg)
			print(rospy.get_time())
			print('')
			rate.sleep()
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
