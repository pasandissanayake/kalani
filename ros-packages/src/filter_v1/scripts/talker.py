#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def talker():
	rospy.init_node('talker', anonymous=True)
	pub = rospy.Publisher('chatter', String, queue_size=10)
	rate = rospy.get_param('~rate',1)
	ros_rate = rospy.Rate(rate)
	rospy.loginfo("Starting ROS node talker....")
	
	while not rospy.is_shutdown():
		msg = "Greetings humans!"
		pub.publish(msg)
		ros_rate.sleep()
		
if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
