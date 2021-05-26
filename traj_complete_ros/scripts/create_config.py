#! /usr/bin/env python

import rospy
import rosparam
import tf.transformations as trans
import math

from geometry_msgs.msg import TransformStamped

if __name__ == "__main__":
    rospy.init_node('config_writer')
    # camera_trans = TransformStamped()
    #
    # camera_trans.header.frame_id = 'world'
    # camera_trans.child_frame_id = 'camera'
    # camera_trans.transform.translation.x = 1.0
    # camera_trans.transform.translation.y = 0.0
    # camera_trans.transform.translation.z = 2.0
    #
    # quat = trans.quaternion_from_euler(0.0, math.pi/2.0, 0, axes='rxyz')
    # camera_trans.transform.rotation.x = quat[0]
    # camera_trans.transform.rotation.y = quat[1]
    # camera_trans.transform.rotation.z = quat[2]
    # camera_trans.transform.rotation.w = quat[3]
    #
    # print(camera_trans)
    #
    # print(camera_trans.__repr__())
    # rosparam.set_param('/camera_trans', camera_trans, verbose=True)
    # rosparam.dump_params('../config/vision.cfg', param='/camera_trans', verbose=True)

    test = rosparam.load_file('../config/vision.cfg')
    rosparam.set_param("test", test[0][0].values())
    rosparam.set_param(test[0][0].keys()[0].__str__(), test[0][0].values()[0].__str__())
    print('test')