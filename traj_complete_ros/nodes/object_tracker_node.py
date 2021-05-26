#!/usr/bin/env python
import collections
import math

import rosbag
import rospy
import message_filters
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker


class object_tracker(object):
    def __init__(self, USE_REAL_CAM=True):
        # self._camera_pose = PoseStamped()
        self._bridge = CvBridge()
        self._mesh_pub = rospy.Publisher('/detected_objects', MarkerArray, latch=True, queue_size=1)

        if USE_REAL_CAM:
            try:
                rospy.wait_for_message('/camera/depth/image_rect_raw', Image, 5.0)
                rospy.wait_for_message("/camera/color/image_raw", Image, 5.0)
                self._depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image, queue_size=5)
                self._color_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=5)

                self._image_time_sync = message_filters.ApproximateTimeSynchronizer([self._color_sub, self._depth_sub], 10, slop=1.0/30.0*1.1)
                self._image_time_sync.registerCallback(self.perception_cb)
            except rospy.ROSException as e:
                print(e)
                exit(22)
        else:
            pass

        rospy.spin()

    def perception_cb(self, color, depth):
        assert isinstance(color, Image)
        assert isinstance(depth, Image)
        # self._depth_sub.unregister()
        # self._color_sub.unregister()
        color_img = self._bridge.imgmsg_to_cv2(color, "passthrough").copy()
        depth_img = self._bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough").copy()

        depth_gray = cv2.normalize(depth_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)        # alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F

        # do perception here...

        color_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        thresh_color, color_thrshd = cv2.threshold(color_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        color_canny = cv2.Canny(color_gray, 0.5 * thresh_color, thresh_color)


        # depth
        depth_gauss = cv2.GaussianBlur(depth_gray,(9,9),0)
        depth_gauss_uint8 = np.uint8(depth_gauss * 255)

        thresh_depth, depth_thrshd = cv2.threshold(depth_gauss_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        depth_canny = cv2.Canny(depth_gauss_uint8, 0.5 * thresh_depth, thresh_depth)

        depth_cont_img, cont_depth, hierachy_depth = cv2.findContours(depth_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        depth_dltd = cv2.dilate(depth_canny,None,iterations=3)

        # lets describe a contour

        

        # some visualization example
        ma = MarkerArray()

        m = Marker()
        m.header.frame_id = 'world'  # could also be in the camera frame
        m.header.stamp = rospy.Time.now()
        m.ns = 'detected_object'
        m.action = m.ADD
        m.type = m.MESH_RESOURCE
        m.mesh_resource = 'package://traj_complete_ros/meshes/model.dae'
        m.color.a = 1.0
        m.color.r = 1.0
        m.pose.position.x = 0.6
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        ma.markers.append(m)

        m = Marker()
        m.header.frame_id = 'world'  # could also be in the camera frame
        m.header.stamp = rospy.Time.now()
        m.ns = 'contour'
        m.action = m.ADD
        m.type = m.LINE_STRIP
        m.color.a = 1.0
        m.color.b = 1.0
        m.scale.x = 0.02
        m.pose.position.x = 0.75
        m.pose.position.y = 0.15
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        for i in range(500):
            p = Point()
            alpha = 2* math.pi * i/500.0
            radius = 0.1
            p.x = math.sin(alpha) * radius
            p.y = math.cos(alpha) * radius
            p.z = 0.05

            m.points.append(p)

        ma.markers.append(m)

        self._mesh_pub.publish(ma)


        # print('wait')

def get_test_data():
    def load_frm_bag(filename):
        """Loads specific topics from rosbag and stores data as np.arrays

        Parameters
        ----------
        filename		: string, required
    			    Name of .bag file
        Outputs
        ----------

        """

        bag = rosbag.Bag(filename)

        # topic_name == '/XTION3/camera/depth_registered/points':
        # no_of_msgs = bag.get_message_count(topic_name)

        x_lst = []
        y_lst = []
        z_lst = []
        rgb_lst = []

        # no need to record time for these, assume static image
        pcl_flag = False
        depth_flag = False
        color_img_flag = False
        for topic, msg, t in bag.read_messages():

            # if topic == '/XTION3/camera/depth_registered/points' and pcl_flag == False:
            # for pt in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            # x_lst.append(pt[0])
            # y_lst.append(pt[1])
            # z_lst.append(pt[2])
            # rgb_lst.append(pt[3])
            # pcl_flag = True

            if (
                    topic == '/XTION3/camera/depth/image_rect' or topic == '/XTION3/camera/depth/image_rect/') and depth_flag == False:
                depth_msg = msg
                depth_flag = True

            elif topic == '/XTION3/camera/rgb/image_rect_color/compressed' and color_img_flag == False:
                color_img_msg = msg
                color_img_flag = True

        bag.close()

        return (x_lst, y_lst, z_lst, rgb_lst, depth_msg, color_img_msg)

    exp_filename = 'depth&rgbOnly_circShape_1stFrame'
    data_dict = {exp_filename: 0}

    namedTup = collections.namedtuple('namedTup', 'x y z rgb depth_msg color_img_msg')

    for file_str in data_dict.keys():
        data = load_frm_bag('../../vision_datasets/first_frame/' + file_str + '.bag')
        data_dict[file_str] = namedTup(data[0], data[1], data[2], data[3], data[4], data[5])

    # # https://stackoverflow.com/questions/47751323/get-depth-image-in-grayscale-in-ros-with-imgmsg-to-cv2-python
    bridge = CvBridge()

    # cv_img = bridge.imgmsg_to_cv2(data_dict[exp_filename].depth_msg, '32FC1')
    # load as opencv images from ROS msg
    # depth_img = bridge.imgmsg_to_cv2(data_dict[exp_filename].depth_msg, "passthrough").copy()
    # color_img = bridge.compressed_imgmsg_to_cv2(data_dict[exp_filename].color_img_msg, "passthrough").copy()

    return data_dict[exp_filename].color_img_msg, data_dict[exp_filename].depth_msg

if __name__ == "__main__":
    rospy.init_node('object_tracker_node')

    rospy.sleep(1.0)
    ot = object_tracker()

    # color, depth = get_test_data()
    # ot.perception_cb(color, depth)





    rospy.loginfo('exiting')