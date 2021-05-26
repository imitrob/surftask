#! /usr/bin/env python


import rospy
from geometry_msgs.msg import Point, Quaternion
from traject_msgs.msg import PointVel
import numpy as np


if __name__ == "__main__":
    rospy.init_node('send_vel')
    rospy.sleep(0.5)
    pub = rospy.Publisher('/motion_goal', PointVel, queue_size=1)
    rospy.sleep(1.0)

    pvel = PointVel()
    rate = rospy.Rate(100)
    start = rospy.Time.now()
    while pvel.pose.position.x < 0.6:
        dur = (rospy.Time.now() - start).to_sec()
        pvel.ang_vel_xyz = np.zeros(3)
        pvel.lin_vel = np.zeros(3)
        pvel.lin_vel[0] = 0.03
        pvel.lin_vel[1] = 0.05*np.cos(dur%(2*np.pi))

        pvel.pose.position = Point(0.2 + dur * 0.03, 0.05*np.sin(dur%(2*np.pi)), 0.4)
        pvel.pose.orientation = Quaternion(0, 1, 0, 0)


        pub.publish(pvel)
        rate.sleep()


    rospy.sleep(2.0)


    # send goal around a circle