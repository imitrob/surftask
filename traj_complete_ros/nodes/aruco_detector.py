#!/usr/bin/env python
import rospy
import message_filters
import cv2
from cv_bridge import CvBridge
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import numpy as np
from time import time


def make_vector3(tvec, order="xyz"):
    return Vector3(**{c: tt for c, tt in zip(map(str, order), np.ravel(tvec).astype(np.float32).tolist())})


def make_quaternion(quat, order="xyzw"):
    return Quaternion(**{c: tt for c, tt in zip(map(str, order), np.ravel(quat).astype(np.float32).tolist())})


def getTransformFromTF(tf_msg):
    trans = np.r_["0,2,0", [getattr(tf_msg.transform.translation, a) for a in "xyz"]]
    rot = tf.transformations.quaternion_matrix(np.r_[[getattr(tf_msg.transform.rotation, a) for a in "xyzw"]])[:3, :3]
    return trans, rot


class Detector(object):
    markerLength = 0.050  # mm
    squareLength = 0.060  # mm
    squareMarkerLengthRate = squareLength / markerLength
    try:
      dictionary = cv2.aruco.Dictionary_create(48, 4, 65536)
    except:
      dictionary = cv2.aruco.Dictionary_create(48, 4)

    def __init__(self):
        self.info_topic = rospy.get_param("/camera_info_topic", '/camera/color/camera_info')
        self.color_topic = rospy.get_param("/color_topic", '/camera/color/image_raw')
        self.optical_frame = rospy.get_param("/optical_frame", 'camera_color_optical_frame')
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber(self.info_topic, CameraInfo, self.camera_info_cb)
        self.tf_publisher = rospy.Publisher("/tf", TFMessage, queue_size=10)

    def camera_info_cb(self, msg):
        self.intrinsics = np.reshape(msg.K, (3, 3))
        self.distCoeffs = np.r_[msg.D]
        self.color_sub = rospy.Subscriber(self.color_topic, Image, self.image_cb)

    def image_cb(self, msg):
        c_image = self.bridge.imgmsg_to_cv2(msg)

        image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)
        color_K = self.intrinsics
        distCoeffs = self.distCoeffs
        markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(image, self.dictionary, cameraMatrix=color_K)

        if len(markerCorners) > 3:
            # print(np.shape(markerCorners))
            diamondCorners, diamondIds = cv2.aruco.detectCharucoDiamond(image, markerCorners, markerIds, self.squareMarkerLengthRate, cameraMatrix=color_K)
            # print(diamondIds)
            # print(diamondCorners)

            if diamondIds is not None and len(diamondIds) > 0:
                img_out = cv2.aruco.drawDetectedMarkers(c_image, markerCorners, markerIds)
                img_out = cv2.aruco.drawDetectedDiamonds(img_out, diamondCorners, diamondIds)
                cv2.imshow("computed markers", img_out)
                cv2.waitKey(1)
                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(np.reshape(diamondCorners, (-1, 4, 2)), self.squareLength, color_K, distCoeffs)
                rmat = cv2.Rodrigues(rvec)[0]
                rmat = np.hstack((rmat, tvec[0].T))
                rmat = np.vstack((rmat, np.r_[0, 0, 0, 1]))
                # rmat = np.linalg.pinv(rmat)
                quat = tf.transformations.quaternion_from_matrix(rmat)

                now = rospy.Time.now()
                trans_msg = TransformStamped()
                trans_msg.header.stamp = now
                # trans_msg.header.frame_id = "marker_frame"
                trans_msg.header.frame_id = self.optical_frame
                # trans_msg.child_frame_id = self.optical_frame
                trans_msg.child_frame_id = "marker_frame"
                trans_msg.transform.translation = make_vector3(tvec)
                trans_msg.transform.rotation = make_quaternion(quat, order="xyzw")
                print(trans_msg)
                tf_msg = TFMessage()
                tf_msg.transforms.append(trans_msg)

                self.tf_publisher.publish(tf_msg)
            else:
                rospy.logwarn("Markers detected but could not detect the diamond.")
        else:
            if len(markerCorners) == 0:
                rospy.logwarn("No markers detected")
            else:
                rospy.logwarn("Detected only {}".format(len(markerCorners)))


if __name__ == "__main__":
    rospy.init_node("camera_marker_own_publisher")
    det = Detector()
    rospy.spin()
