#!/usr/bin/env python
import math

import rospy
import rospkg
import message_filters
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Vector3, Point, TransformStamped
from traject_msgs.msg import ContourArrayStamped
import actionlib
from traject_msgs.msg import CurveExecutionAction, CurveExecutionGoal, CurveExecutionResult, CurveExecutionFeedback
import numpy as np
import os
import argparse
import yaml
from traj_complete_ros.utils import contour_msg2list, ftl_pcl2numpy, getTransformFromTF
import open3d as o3d
import tf2_ros
import tf


class UI(object):

    def __init__(self, config, nogui=False, show_pcl=False):

        self.config = config


        self.curve = None
        self.xyz, self.rgb = None, None
        self.custom_contour = None
        self.is_drawing = False
        self.is_executing = False
        self.window_name = 'contour image'
        self.action_client = actionlib.SimpleActionClient('curve_executor', CurveExecutionAction)

        try:
            self.action_client.wait_for_server(rospy.Duration(20))  # wait till the action server is up

            self.rate = rospy.Rate(30)  # rate for the main loop

            rospy.on_shutdown(self.destroy)
        except rospy.ROSException as e:
            print(e)
            exit(22)
        else:

            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)



    def applyCurve(self):
        """ Apply curve to the selected contour. The contour will be stored in "current_contour" variable.
        """
        if self.custom_contour is not None:  # if there is a custom contour, select that by default
            current_contour = self.custom_contour
        elif self.selected_contour >= 0:  # otherwise if some other contour is selected, use that
            current_contour = self.contours[self.selected_contour]
        else:
            rospy.logwarn("No contour selected, cannot apply curve!")
            return

        # for now, apply random noise
        m = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
        cv2.fillPoly(m, [current_contour + np.random.randint(-10, 10, current_contour.shape)], 255, 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
        self.curve, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.curve = self.curve[-1]
        self.redrawContours()


    def compute3DCurve(self):
        """ For a generated 2D curve, computes the corresponding 3D points and estimates normals for them.
        """
        trans, rmat = getTransformFromTF(self.depth_to_color_tf)
        points_in_color_frame = np.dot(rmat, self.xyz.T) + trans
        self.pcd.points = o3d.utility.Vector3dVector(points_in_color_frame.T)
        self.pcd.normals = o3d.utility.Vector3dVector(np.ones_like(self.xyz))
        rgb_as_f = self.rgb.astype(np.float) / 255  # normalize to <0, 1> range
        self.pcd.colors = o3d.utility.Vector3dVector(rgb_as_f)
        self.pcd.orient_normals_towards_camera_location()  # need to pre-orient normals to remove ambiguity in the normal estimation
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)

        zilch = np.zeros((3, 1), dtype=np.float32)
        points_2d, _ = cv2.projectPoints(np.array(self.pcd.points), zilch, zilch, self.camera_matrix, self.distCoeffs)
        points_2d = np.clip(np.floor(points_2d.squeeze()).astype(np.int16), (0, 0), (self.cam_intrinsics.width - 1, self.cam_intrinsics.height - 1))  # it's x, y!

        mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
        cv2.drawContours(mask, [self.curve], 0, 255, 1)
        contour_pt_idx = np.where(mask[points_2d[..., 1], points_2d[..., 0]] == 255)[0]

        self.curve_pcd = self.pcd.select_down_sample(contour_pt_idx)
        self.current_curve_len = np.array(self.curve_pcd.points).shape[0]

        if self.show_pcl:
            # self.curve_pcd.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([self.curve_pcd, self.pcd.select_down_sample(contour_pt_idx, invert=True)])
            o3d.visualization.draw_geometries([self.curve_pcd])

        return np.size(self.curve_pcd.points) > 0

    def sendCurve(self):
        """If there is a pointcloud for a curve, sends it to the controller as goal
        """
        counters = np.linspace(0,2*math.pi,250)
        factors = np.linspace(0.05, 0.05, 250)
        heights = np.linspace(0.1, 0.1, 250)

        curve_points = [[R*np.sin(counter) +0.2, R*np.cos(counter), h] for counter, R, h in zip(counters, factors, heights)]
        curve_normals = [[0,0,1] for _ in range(len(curve_points))]

        self.action_client.send_goal(
            CurveExecutionGoal(
                curve=[Point(**{k: v for k, v in zip(["x", "y", "z"], p)}) for p in np.array(curve_points)],
                normals=[Vector3(**{k: v for k, v in zip(["x", "y", "z"], n)}) for n in np.array(curve_normals)],
            ),
            self.done_cb, self.active_cb, self.feedback_cb
        )

    def change_curveType(self, val):
        """Selects a curve type from a curve library (stored in self.config["curve_library"])

        Args:
            val (int): Value from the trackbar in range 0-<#_of_curves_in_lib>
        """
        pass

    def done_cb(self, success_code, msg):
        """Callback for the curve execution. This is called when the action is done.
        """
        self.is_executing = False
        # cv2.setTrackbarPos("progress", "contour image", 0)
        print("Success code: {}, result: {}".format(success_code, msg.done))

    def active_cb(self):
        """Callback for the curve execution. This is called when the action is started.
        """
        self.is_executing = True
        print("Action time!")

    def feedback_cb(self, msg):
        """Callback for the curve execution. This is called when the action is progressed.
        The message argument contains progress information.
        """
        print(msg)
        # cv2.setTrackbarPos("progress", "contour image", int(np.round(float(msg.progress) / self.current_curve_len * 100)))

    def run(self):
        """This is the main loop function. It first creates the relevant OpenCV window and control components.
        Afterwards, it loops over, redraws the image and listens for user inputs.
        """

        self.sendCurve()



    def destroy(self):
        # called when exiting
        if self.is_executing:
            self.action_client.cancel_goal()
            self.action_client.stop_tracking_goal()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node('interface_node')


    ui = UI(config=None)
    ui.run()

    rospy.loginfo('exiting')
