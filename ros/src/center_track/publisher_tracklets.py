#!/usr/bin/env python
import rospy
from std_msgs.msg import String,Header,Float32MultiArray,MultiArrayDimension
from std_msgs.msg import String,Int32,Int32MultiArray , Float64MultiArray 
from sensor_msgs.msg import Image,PointCloud2,PointField
from sensor_msgs import point_cloud2
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from rospy.numpy_msg import numpy_msg
#from test_publisher.msg import test_msg
import ros_np_multiarray as rnm
#from kivy.clock import Clock
import time
from label_utils import *
import pickle

bridge = CvBridge()

image_file_path = "/media/entc/New Volume/data/2011_09_26/2011_09_26_drive_0022_sync/image_02/data"
velodyne_file_path = "/media/entc/New Volume/data/2011_09_26/2011_09_26_drive_0022_sync/velodyne_points/data"
label_file_path = "/media/entc/New Volume/DATASETS/Detection-KITTI/data_object_label_2/training/label_2"
xml_path ="/media/entc/New Volume/data/2011_09_26/2011_09_26_drive_0022_sync/tracklet_labels.xml"



def GetXMLLabels( xml_path , num_frames):

    tracklet_, type_ = load_tracklet(  xml_path , num_frames= num_frames  )
    print(tracklet_)
    return tracklet_ , type_

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
	print(len(velodynes))
	tracklets , type = load_tracklet( xml_path , len(velodynes) )
	print("Number of tracklets ",len(tracklets))
	#left_imgs2 = left_imgs[2:7]
	pub = rospy.Publisher('left_camera_images', Image, queue_size=0)
	pub2 = rospy.Publisher('point_clouds', PointCloud2, queue_size=0)
	pub3 = rospy.Publisher('label', String, queue_size=0)
	pub4 = rospy.Publisher('tracklets', Float32MultiArray , queue_size=0)
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
			"""
            label_file = open( labels[i] , "r")
            label_data = str( label_file.read() )
            label_msg = String( data = label_data )
			"""  
			ts = rospy.Time.now()
			data_to_send = Float32MultiArray()  # the data to be sent, initialise the array
			if( tracklets[i] is None ):
				print(tracklets[i])
			data = rnm.to_multiarray_f32(np.array( tracklets[i] , dtype=np.float32))
			
			print( np.array(tracklets[i]).shape )
			#data_to_send.data = data # assign the array with the value you want to send

			img = cv2.imread(left_imgs[i])
		
			#image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
			image_message = bridge.cv2_to_imgmsg(img, "bgr8")
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
            #pub3.publish( label_msg )
			pub4.publish(data)
			print(data)
			#pub3.publish(scoresmsg)
			print(rospy.get_time())
			rate.sleep()
			#rospy.sleep(0.1)

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
