#!/usr/bin/env python
from __future__ import print_function, division
from scipy.interpolate import interpolate
import rospy
import rospkg
import message_filters
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Vector3, Point, Pose, TransformStamped, Transform, Quaternion
from traject_msgs.msg import ContourArrayStamped, ContourPrimitive, ExecutionOptions, LoggingTask
from visualization_msgs.msg import MarkerArray, Marker
import actionlib
from traject_msgs.msg import CurveExecutionAction, CurveExecutionGoal
from std_msgs.msg import ColorRGBA
import numpy as np
import os
import argparse
import yaml
from traj_complete_ros.geometry import R_axis_angle
from traj_complete_ros.utils import contour_msg2list, ftl_pcl2numpy, ftl_numpy2pcl, getTransformFromTF, BsplineGen
from traj_complete_ros.LoggerProxy import LoggerProxy
from traj_complete_ros.apply_pattern_to_contour import apply_pattern_to_contour, ApproxMethod
import open3d as o3d
import tf2_ros
import tf.transformations as tftrans
from scipy import signal
from uuid import uuid4
import sys
import time
import traceback


class UI(object):

    CURVE_UPSAMPLING_FACTOR = 2

    def __init__(self, config, no_wait=False, show_pcl=False, full_pcl=False, original_pcl=False):
        color_topic = rospy.get_param("/color_topic", '/camera/color/image_raw')
        camInfo_topic = rospy.get_param("/color_info_topic", '/camera/color/camera_info')
        pcl_topic = rospy.get_param("/pcl_topic", '/camera/depth/color/points')
        contour_topic = rospy.get_param("/contour_topic", '/contour/raw')
        self.color_frame_id = rospy.get_param("/color_frame_id", "camera_color_optical_frame")
        self.pcl_frame_id = rospy.get_param("/pcl_frame_id", "camera_depth_optical_frame")
        self.table_frame_id = rospy.get_param("/table_frame_id", 'camera_link')
        # self.table_frame_id = rospy.get_param("/table_frame_id", 'table1')
        self.show_pcl = show_pcl
        self.show_full_pcl = full_pcl
        self.show_orig_pcl = original_pcl

        # load config data and parameters
        self.config = config
        self.patternChoices = list(self.config["curve_library"].keys())
        self.patternType = self.config["default_pattern"] if self.config["default_pattern"] in self.patternChoices else self.patternChoices[0]
        self.z_comp = self.config["z_computation"]
        self.num_pattern_reps = self.config["num_of_reps"]
        self.approx_method = ApproxMethod.fromString(self.config["approx_method"])
        self.c_smoothing_enabled = self.config["c_smoothing_enabled"]
        self.c_smoothing_sigma = self.config["c_smoothing_sigma"]
        self.c_smoothing_kernel_size = self.config["c_smoothing_kernel_size"]
        self.pattern_gap = self.config["pattern_gap"]
        self.pattern_rotation = self.config["pattern_rotation"]
        print('rotation {}'.format(self.pattern_rotation))
        _, self.exec_engine = self._convert_exec_engine(self.config["executor_engine"])
        self.cart_speed = self.config["cart_vel_limit"]
        self.pattern_trim_start = self.config["pattern_trim_start"]
        self.pattern_trim_trail = self.config["pattern_trim_trail"]
        self.pcl_interpolation = self.config["pcl_interpolation"]

        self._bridge = CvBridge()
        self.pauseUpdate = False
        self.kernelDilate = np.ones((3, 3), np.uint8)
        self.kernelClose = np.ones((3, 3), np.uint8)
        self.kernelOpen = np.ones((5, 5), np.uint8)
        self.contour_scale = 0
        self.contour_offsetX = 0
        self.contour_offsetY = 0
        self.selected_contour = -1
        self.curve = None
        self.xyz, self.rgb = None, None
        self.custom_contour = None
        self.is_drawing = False
        self.is_executing = False
        self.window_name = 'contour image'
        self.action_client = actionlib.SimpleActionClient('curve_executor', CurveExecutionAction)
        self.last_update = None
        self.is_done = False
        self.curve_pcd = None
        self.last_execution_result = False

        self.rate = rospy.Rate(30)  # rate for the main loop
        rospy.on_shutdown(self.destroy)
        self.pcd = o3d.geometry.PointCloud()
        self.contours = []

        if no_wait:
            self.cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(848, 480, *np.r_[601.640869140625, 0.0, 419.38916015625, 0.0, 601.1117553710938, 245.51934814453125, 0.0, 0.0, 1.0][[0, 4, 2, 5]])
            self.camera_matrix = np.reshape([601.640869140625, 0.0, 419.38916015625, 0.0, 601.1117553710938, 245.51934814453125, 0.0, 0.0, 1.0], (3, 3))
            self.distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.input_image = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width, 3), dtype=np.uint8)
        else:
            try:
                img_msg = rospy.wait_for_message(color_topic, Image, 15.0)
                cameraInfo_msg = rospy.wait_for_message(camInfo_topic, CameraInfo, 15.0)
                rospy.wait_for_message(contour_topic, ContourArrayStamped, 15.0)  # wait for detection to come online
            except rospy.ROSException as e:
                print(e)
                exit(22)

            self._color_sub = message_filters.Subscriber(color_topic, Image, queue_size=200)
            self._contour_sub = message_filters.Subscriber(contour_topic, ContourArrayStamped, queue_size=200)
            self._pcl_sub = message_filters.Subscriber(pcl_topic, PointCloud2, queue_size=200)
            self.tf_buffer = tf2_ros.Buffer(rospy.Duration.from_sec(20))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            # self._image_time_sync = message_filters.ApproximateTimeSynchronizer([self._color_sub, self._contour_sub], 200, slop=0.1)
            self._image_time_sync = message_filters.ApproximateTimeSynchronizer([self._color_sub, self._contour_sub, self._pcl_sub], 100, slop=0.5)
            self._image_time_sync.registerCallback(self.perception_cb)

            self.cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(img_msg.width, img_msg.height, *np.r_[cameraInfo_msg.K][[0, 4, 2, 5]])
            self.camera_matrix = np.reshape(cameraInfo_msg.K, (3, 3))
            self.distCoeffs = np.array(cameraInfo_msg.D)
            self.input_image = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width, 3), dtype=np.uint8)
        try:
            self.action_client.wait_for_server(rospy.Duration(20))  # wait till the action server is up
        except rospy.ROSException as e:
            print(e)
            exit(22)

        self.pcl_pub = rospy.Publisher("curve_pcl", PointCloud2, queue_size=10, latch=True)
        self.contour_image = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width, 3), dtype=np.uint8)
        self.info_panel = np.zeros((56, self.cam_intrinsics.width, 3), dtype=np.uint8)
        self.vis_pub = rospy.Publisher("marker", Marker, queue_size=10, latch=True)
        # self.vis_mul_pub = rospy.Publisher("marker_array", MarkerArray, queue_size=10, latch=True)

    @staticmethod
    def _convert_exec_engine(val):
        """Takes either int or string and returns a tuple of (str, int),
        where str is the name of the execution engine and int is its "id"
        """
        compare = lambda k, v: k.lower() == val.lower() if type(val) is str else v == val  # noqa
        try:
            return next((k, v) for k, v in ExecutionOptions.__dict__.items() if not k.startswith('_') and not callable(getattr(ExecutionOptions, k)) and type(v) is int and compare(k, v))
        except StopIteration:
            return ("MOVEIT_CART_PLANNING", 1)  # default engine

    def adjust_contours(self, val):
        # contour scaling trackbar cb
        self.contour_scale = val - 128
        self.recalcContours()

    def adjust_offsetX(self, val):
        # X offset trackbar cb
        self.contour_offsetX = val - 128
        self.recalcContours()

    def adjust_offsetY(self, val):
        # Y offset trackbar cb
        self.contour_offsetY = val - 128
        self.recalcContours()

    def mouse_cb(self, event, x, y, flags, params):
        # mouse event cb
        if event == cv2.EVENT_LBUTTONUP:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:  # contour selection
                self.selectContour(x, y)
            if self.is_drawing:  # if drawing stop drawing
                self.is_drawing = False
                self.custom_contour = np.array(self.custom_contour)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:  # start drawing
                self.custom_contour = []
                self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:  # if drawing, store current mouse position
                self.custom_contour.append([x, y])
                self.redrawContours()
        elif event == cv2.EVENT_MBUTTONUP:  # apply curve
            if not self.is_drawing:
                self.applyCurve()
            else:
                rospy.logwarn("Currently drawing a custom contour, cannot apply curve!")

    def _drawCTlineByline(self, contour):
        image = np.ones((self.cam_intrinsics.height, self.cam_intrinsics.width, 3), dtype=np.uint8) * 255
        contour = contour.squeeze()
        for i, (x, y) in enumerate(contour):
            x, y = int(x), int(y)
            if i == 0:
                image = cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            else:
                image = cv2.line(image, tuple(contour[i - 1].astype(np.int).tolist()), (x, y), (0, 0, 255))

        cv2.imshow("contour", image)
        cv2.waitKey(0)

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
        # m = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
        # cv2.fillPoly(m, [current_contour + np.random.randint(-10, 10, current_contour.shape)], 255, 1)
        # m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
        # self.curve, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.curve = self.curve[-1]

        n_points = current_contour.shape[0] * self.CURVE_UPSAMPLING_FACTOR
        #current_contour.shape[0] * self.CURVE_UPSAMPLING_FACTOR

        bsp = BsplineGen.fromLibraryData(**self.config["curve_library"][self.patternType])
        print('self.patternType: {}'.format(self.patternType))
        n_points = max([bsp.nmbPts * self.num_pattern_reps, 2*current_contour.shape[0], 200])
        try:
            contour_splined = np.array(list(reversed(BsplineGen.fromWaypoints(current_contour).generate_bspline_sample(n_points))))
        except:
            contour_splined = current_contour
        applied_bsp = apply_pattern_to_contour(bsp, contour_splined, self.num_pattern_reps,
                                               approx_method=self.approx_method, pattern_rotation=self.pattern_rotation,
                                               pattern_trim_start=self.pattern_trim_start,
                                               pattern_trim_trail=self.pattern_trim_trail,
                                               use_gap=self.pattern_gap)
        self.curve = applied_bsp.generate_bspline_sample(n_points)[
            :, np.newaxis, :]
        # self.curve = np.round(applied_bsp.generate_bspline_sample(n_points)[:, np.newaxis, :]).astype(np.int)
        # self._drawCTlineByline(self.curve)
        self.redrawContours()

    def select_contour_pars(self, BsplineGen):
        # TODO: user should see the given pattern applied along line and along given contour (circle?)
        # TODO: user selects parameters - trimming, rotation, gap between patterns to connect them (this maybe is set automatically)
        # TODO: these parameters are saved along with the bspline to the library
        print('to be implemented')
        return

    def selectContour(self, x, y):
        """Tries to select a contour by position. The x, y coordinates are of a point (typically from a mouse click)
        that is used to locate the contour. If the point lies inside a contour, that contour is selected.

        Args:
            x (int)
            y (int)
        """
        pt = (x, y)
        dists = np.r_[[cv2.pointPolygonTest(c, pt, measureDist=True) for c in self.contours]]
        dists[np.where(dists < 0)] = np.inf
        if np.all(np.isposinf(dists)):
            rospy.loginfo("The mouse was not clicked inside any contour, cannot select!")
            return

        sel = np.argmin(dists)
        if sel.size == 1:
            sel = int(sel)
            if sel == self.selected_contour:
                self.selected_contour = -1  # deselect
            else:
                self.selected_contour = sel  # select
            self.redrawContours()
        else:
            rospy.logwarn("Cannot select contour, for some reason :-O")

    def cycleSelectContour(self):
        """Cyclically selects the individual contours, starting with the larges one.
        """
        if self.selected_contour > -1:  # some contour is already selected
            self.selected_contour += 1  # select next index
            if self.selected_contour >= len(self.contours):
                self.selected_contour = 0
        else:  # not contour is currently selected
            if len(self.contours) > 0:
                self.selected_contour = np.argmax([c.shape[0] for c in self.contours])  # select the largest one
            else:
                rospy.logwarn("No contours to select!")
        self.redrawContours()

    def compute3DCurve(self):
        """ For a generated 2D curve, computes the corresponding 3D points and estimates normals for them.
        """
        # TODO: add option to compute "flat 3D" curve
        # TODO: pattern "drop-of-edge" warning
        # get transformation from PCL to RGB image
        trans, rmat = getTransformFromTF(self.depth_to_color_tf)
        points_in_color_frame = np.dot(rmat, self.xyz.T) + trans
        # points_in_color_frame = self.xyz.T

        # create Open3D PCL from numpy PCL
        self.pcd.points = o3d.utility.Vector3dVector(points_in_color_frame.T)
        self.pcd.normals = o3d.utility.Vector3dVector(np.ones_like(self.xyz))
        rgb_as_f = self.rgb.astype(np.float) / 255  # normalize to <0, 1> range
        self.pcd.colors = o3d.utility.Vector3dVector(rgb_as_f[..., ::-1])

        # compute normals
        self.pcd.orient_normals_towards_camera_location()  # need to pre-orient normals to remove ambiguity in the normal estimation
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)

        # PCL interp
        if self.pcl_interpolation:
            # d = self.pcd.compute_nearest_neighbor_distance()
            # med = np.median(d)
            # std = np.std(d)
            # radii = [np.min(d), med - std, med, med + std, np.max(d)]
            # tmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.pcd, o3d.utility.DoubleVector(radii))
            tmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, 7)
            pcd = tmesh.sample_points_uniformly(int(1e6))
            # o3d.visualization.draw_geometries([pcl])
        else:
            pcd = self.pcd

        # project PCL to 2D
        zilch = np.zeros((3, 1), dtype=np.float32)
        points_2d, _ = cv2.projectPoints(np.array(pcd.points), zilch, zilch, self.camera_matrix, self.distCoeffs)
        # clip points projected outside of the image
        # points_2d = np.clip(np.floor(points_2d.squeeze()).astype(np.int16), (0, 0), (self.cam_intrinsics.width - 1, self.cam_intrinsics.height - 1))  # it's x, y!
        points_2d = points_2d.squeeze()

        # # create mask from curve and select points that lie on the mask
        # mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
        # cv2.drawContours(mask, [self.curve], 0, 255, 1)
        # contour_pt_idx = np.where(mask[points_2d[..., 1], points_2d[..., 0]] == 255)[0]

        # contour_pt_idx = [np.argmin((np.abs(points_2d - cpt)).sum(axis=1)) for cpt in self.curve.squeeze()]
        curve = self.curve.squeeze()
        pcl = np.array(pcd.points)
        diffs = [np.linalg.norm(points_2d - cpt, axis=1) for cpt in curve]
        curve_3d = np.full((curve.shape[0], 3), np.inf, dtype=np.float)
        for i, (df, p) in enumerate(zip(diffs, self.curve.squeeze())):
            close = np.where(df < 2)[0]
            w = df[close].ravel()
            if np.sum(w) == 0:
                continue
            w = np.abs(1 - (w / w.sum() + 1e-10))
            curve_3d[i, :] = np.average(pcl[close, :], weights=w, axis=0)

        curve_3d = curve_3d[np.logical_not(np.any(np.isinf(curve_3d), axis=1)), :]
        if self.c_smoothing_enabled:
            half = self.c_smoothing_kernel_size >> 1
            curve_3d_padded = np.vstack((
                curve_3d[-half:, :],
                curve_3d,
                curve_3d[:(self.c_smoothing_kernel_size - half - 1), :]
            ))
            gkernel = np.tile(signal.gaussian(self.c_smoothing_kernel_size, self.c_smoothing_sigma)[:, np.newaxis], (1, 3))
            # gkernel /= gkernel.sum()
            curve_3d_smooth = signal.fftconvolve(curve_3d_padded, gkernel, mode="valid", axes=0)
            ret, M, _ = cv2.estimateAffine3D(curve_3d_smooth, curve_3d)
            if ret:
                curve_3d = np.dot(M, cv2.convertPointsToHomogeneous(curve_3d_smooth).squeeze().T).T

        # if True:
        # # if self.c_resamp_enabled:
        #     padding_amount = 12
        #     up_samp_amount, down_samp_amount = 2, 4
        #     data_resamp = signal.resample_poly(np.pad(curve_3d, ((padding_amount, padding_amount), (0, 0)), mode="edge"), up_samp_amount, down_samp_amount, axis=0, window=("kaiser", 10))
        #     unpad_amount = int(padding_amount * (up_samp_amount / down_samp_amount))
        #     curve_3d = data_resamp[unpad_amount:-unpad_amount, :]

        # self.curve_pcd = self.pcd.select_down_sample(contour_pt_idx)
        # fake_points = np.hstack((self.curve.squeeze(), np.ones((self.curve.shape[0], 1))))
        self.curve_pcd = o3d.geometry.PointCloud()
        self.curve_pcd.points = o3d.utility.Vector3dVector(curve_3d)
        self.curve_pcd.paint_uniform_color([1, 0.0, 0.2])
        self.curve_pcd.normals = o3d.utility.Vector3dVector(np.ones_like(curve_3d))
        self.curve_pcd.orient_normals_towards_camera_location()  # need to pre-orient normals to remove ambiguity in the normal estimation
        self.curve_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)
        # self.curve_pcd.points = o3d.utility.Vector3dVector(np.array(self.pcd.points)[contour_pt_idx, :])
        # self.curve_pcd.normals = o3d.utility.Vector3dVector(np.array(self.pcd.normals)[contour_pt_idx, :])
        # self.curve_pcd.colors = o3d.utility.Vector3dVector(np.array(self.pcd.colors)[contour_pt_idx, :])
        self.current_curve_len = np.array(self.curve_pcd.points).shape[0]

        abbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(self.curve_pcd.points)
        abbox.scale(2, True)
        cropped_pcd = pcd.crop(abbox)
        cropped_pcd_orig = self.pcd.crop(abbox)

        self.curve_points, self.curve_normals = np.array(self.curve_pcd.points), np.array(self.curve_pcd.normals)
        # self.curve_points, self.curve_normals = self.publish_pcl(self.curve_pcd, cropped_pcd)

        if self.show_pcl or self.show_full_pcl or self.show_orig_pcl:
            # self.curve_pcd.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([self.curve_pcd, self.pcd.select_down_sample(contour_pt_idx, invert=True)])

            n = self.current_curve_len
            a = np.arange(0, n - 1)
            b = np.arange(1, n)
            correspondences = [(i, j) for (i, j) in np.stack((a, b)).T]
            lineSet = o3d.geometry.LineSet.create_from_point_cloud_correspondences(self.curve_pcd, self.curve_pcd, correspondences)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Curve visualization", width=1920, height=1012)
            # vis.lookat()
            ctr = vis.get_view_control()

            vis.add_geometry(lineSet)
            if self.show_orig_pcl:
                vis.add_geometry(cropped_pcd_orig)
            elif self.show_full_pcl:
                vis.add_geometry(cropped_pcd)
            if self.show_pcl:
                vis.add_geometry(self.curve_pcd)

            # param = o3d.io.read_pinhole_camera_parameters("/home/syxtreme/tan_ws/src/icra-2021-trajectory-autocompletion-tan-thesis/traj_complete_ros/config/viewpoint.json")
            # ctr.convert_from_pinhole_camera_parameters(param)
            ctr.change_field_of_view(5.0)
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters("/home/syxtreme/tan_ws/src/icra-2021-trajectory-autocompletion-tan-thesis/traj_complete_ros/config/viewpoint3.json", param)
            vis.destroy_window()
        return np.size(self.curve_pcd.points) > 0

    def publish_pcl(self, curve_pcd, object_pcd):
        curve = np.pad(np.array(curve_pcd.points), ((0, 0), (0, 1)), mode="constant", constant_values=1)
        object_pts = np.pad(np.array(object_pcd.points), ((0, 0), (0, 1)), mode="constant", constant_values=1)
        normals = np.array(curve_pcd.normals)
        # post processing for curves without camera calibration
        curve = curve - curve.mean(axis=0) + np.array([0.0, 0.0, 0.0, 1.0])
        object_pts = object_pts - object_pts.mean(axis=0) + np.array([0.0, 0.0, 0.0, 1.0])
        # curve = curve - curve.mean(axis=0) + np.array([0.1, 0.0, 0.2, 1.0])
        # v1*v2 = | v1 | | v2 | cos(angle)
        normal_avg = np.mean(normals, axis=0)
        v2 = np.array([0, 0, 1])
        # assumes vectors to be normalized!!!
        angle = np.arccos(np.dot(v2, normal_avg))
        axis = np.cross(v2, normal_avg)
        R = np.eye(3)
        R_axis_angle(R, axis=axis, angle=angle)
        shift = np.eye(4)
        shift[:3, :3] = R
        curve_new = np.dot(curve, shift)[:, :3] + np.array([0.1, 0.0, 0.2])
        normals_new = np.dot(normals, R)
        object_new = np.dot(object_pts, shift)[:, :3] + np.array([0.1, 0.0, 0.17])

        header = Header(stamp=rospy.Time.now(), frame_id="table2")

        # xyz = np.hstack((object_new.T, curve_new.T))
        # rgb = np.hstack((np.array(object_pcd.colors).T, np.array(curve_pcd.colors).T))
        xyz = object_new.T
        rgb = np.array(object_pcd.colors).T

        pcl_msg = ftl_numpy2pcl(xyz, header, rgb3d=(rgb.T * 255).astype(np.uint8))
        self.pcl_pub.publish(pcl_msg)

        return curve_new, normals_new

    def composeArtificialCurve(self, name, n_points=1024, libPath="config/hand_trajectories.yaml"):
        curve = BsplineGen.loadPatternFromLibrary(os.path.join(rospack.get_path('traj_complete_ros'), libPath), name)
        if curve is None:
            return False

        self.curve_points = np.hstack((curve.generate_bspline_sample(n_points), np.ones((n_points, 1)) * 0.05))
        self.curve_normals = np.hstack((np.zeros((n_points, 2)), -np.ones((n_points, 1))))
        self.curve_pcd = o3d.geometry.PointCloud()
        self.curve_pcd.points = o3d.utility.Vector3dVector(self.curve_points)
        self.curve_pcd.paint_uniform_color([1, 0.0, 0.2])
        self.curve_pcd.normals = o3d.utility.Vector3dVector(self.curve_normals)
        self.current_curve_len = np.array(self.curve_pcd.points).shape[0]

    def sendCurve(self, log_name=""):
        """If there is a pointcloud for a curve, sends it to the controller as goal
        """
        from datetime import datetime
        log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.curve_pcd is None:
            rospy.logwarn("The curve pointcloud hasn't been computed, yet. Cannot send goal to the controller.")
            return

        timestamp = rospy.Time.now()
        trans, rmat = getTransformFromTF(self.color_to_table_tf)

        c_points = (np.dot(rmat, self.curve_points.T) + trans).T + np.r_[0, 0, -0.011]
        c_normals = np.dot(rmat, self.curve_normals.T).T

        # TODO: this was for real exps, I guess and it should be removed.
        # c_points[:,2] = 0.035

        self.is_done = False
        self.last_execution_result = False
        exec_options = ExecutionOptions(
            executor_engine=self.exec_engine,
            cart_vel_limit=self.cart_speed
        )

        # publish trajectory for visualization
        marker = Marker(
            header=Header(
                stamp=timestamp,
                frame_id=self.table_frame_id
            ),
            # pose=Pose(
            #     position=Point(**{k: v for k, v in zip(["x", "y", "z"], p)}),
            #     # orientation=Vector3(**{k: v for k, v in zip(["x", "y", "z"], n)}),
            # ),
            id=10001,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            points=[Point(**{k: v for k, v in zip(["x", "y", "z"], p)}) for p in c_points],
            lifetime=rospy.Duration(0),
            scale=Vector3(x=0.001, y=0.001, z=0.001),
            color=ColorRGBA(r=0.1, g=0.9, b=0.7, a=0.8)
        )
        self.vis_pub.publish(marker)

        exec_options.executor_engine = ExecutionOptions.DESCARTES_TOLERANCED_PLANNING

        goal = CurveExecutionGoal(
                header=Header(frame_id="table2"),
                # header=Header(frame_id=self.table_frame_id),
                opt=exec_options,
                curve=[Point(**{k: v for k, v in zip(["x", "y", "z"], p)}) for p in c_points],
                normals=[Vector3(**{k: v for k, v in zip(["x", "y", "z"], n)}) for n in c_normals],
            )
        if log_name:
            LoggerProxy.logger(action=LoggingTask.RESET)
            LoggerProxy.logger(action=LoggingTask.SETREFERENCE, data=goal)
            LoggerProxy.logger(action=LoggingTask.START)

            def my_done_cb(success_code, msg):
                LoggerProxy.logger(action=LoggingTask.STOP)
                if success_code == actionlib.GoalStatus.SUCCEEDED:
                    # fp = "/home/behrejan/traj_complete_log/" + log_name + ".json"
                    fp = os.path.expanduser('~/traj_complete_log/') + log_name + ".json"
                    LoggerProxy.logger(action=LoggingTask.SAVE, data=fp)
                else:
                    print('Execution failed, not saving the log!')
                self.done_cb(success_code, msg)
            done_cb = my_done_cb
        else:
            done_cb = self.done_cb

        self.action_client.send_goal(
            goal,
            done_cb, self.active_cb, self.feedback_cb
        )

    def change_curveType(self, val):
        self.patternType = self.patternChoices[val]

    def change_approxMethod(self, val):
        self.approx_method = 2**val

    def change_z_computation(self, val):
        self.z_comp = (val and "flat") or "real"

    def change_c_smooth(self, val):
        self.c_smoothing_enabled = bool(val)

    def change_exec_engine(self, val):
        _, val = self._convert_exec_engine(val)  # let's check if choice is valid
        if val == 0:
            val = 1  # this engine is not implemented, yet
        cv2.setTrackbarPos("exec engine", self.window_name, val)
        self.exec_engine = val

    def change_cart_speed(self, val):
        if val < 1:
            val = 1  # this engine is not implemented, yet
            cv2.setTrackbarPos("cart speed (cm/s)", self.window_name, val)
        self.cart_speed = float(val) / 100  # convert from cm/s -> m/s

    def set_numOfReps(self, val):
        if val < 1:
            val = 1
            cv2.setTrackbarPos("num of reps", self.window_name, 1)
        self.num_pattern_reps = val

    def change_pattern_rotation(self, val):
        self.pattern_rotation = val - 180

    def change_pattern_gap(self, val):
        self.pattern_gap = val - 128

    def change_pcl_interp(self, val):
        self.pcl_interpolation = bool(val)

    def change_show_orig_pcl(self, val):
        self.show_orig_pcl = bool(val)

    def change_trim_start(self, val):
        self.pattern_trim_start = val

    def change_trim_trail(self, val):
        self.pattern_trim_trail = val

    def done_cb(self, success_code, msg):
        """Callback for the curve execution. This is called when the action is done.
        """
        self.is_executing = False
        self.is_done = True
        self.last_execution_result = success_code == actionlib.GoalStatus.SUCCEEDED
        try:
            cv2.setTrackbarPos("progress", "contour image", 0)
        except:  # noqa
            pass
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
        try:
            cv2.setTrackbarPos("progress", "contour image", int(float(msg.progress)))  # expects percentage
        except:  # noqa
            pass

    def selectAndSendCurve(self, save=False, log_name=""):
        while self.last_update is None:
            self.rate.sleep()
        self.pauseUpdate = True
        self.contour_scale = -20
        self.contour_offsetY = -4
        self.recalcContours(False)
        self.selected_contour = np.argmax([c.shape[0] for c in self.contours])
        self.applyCurve()
        self.compute3DCurve()

        if save:  # if save -> save curve and don't send
            np.savez("curve_{}".format(str(uuid4())), points=np.array(self.curve_pcd.points), curve_pcd=self.curve_pcd)
            return

        self.sendCurve(log_name=log_name)

        while not self.is_done:
            self.rate.sleep()
        rospy.signal_shutdown("I am done!")

    def SendTestCurve(self, curve_primitive,
                      exec_options=ExecutionOptions(executor_engine=ExecutionOptions.MOVEIT_CART_PLANNING),
                      log_name="", pattern=None):
        '''
        This function builds and sends a contour msg from a curve primitive. Main usage: test cases for the execution
        :param curve_primitive: curve curve_primitive msg
        :return: None
        '''
        nr_points = 250
        assert isinstance(curve_primitive, ContourPrimitive)
        if curve_primitive.type == ContourPrimitive.SPIRAL:
            if curve_primitive.ccw:
                counters = np.linspace(0, curve_primitive.rounds * 2 * np.pi, nr_points)
            else:
                counters = np.linspace(0, -curve_primitive.rounds * 2 * np.pi, nr_points)
            factors = np.linspace(curve_primitive.r_start, curve_primitive.r_end, nr_points)
            heights = np.linspace(curve_primitive.h_start, curve_primitive.h_end, nr_points)

            curve_points = np.array([[R * np.sin(counter), R * np.cos(counter), h] for counter, R, h in zip(counters, factors, heights)])
            curve_normals = np.array([curve_primitive.normal for _ in range(nr_points)])
        if curve_primitive.type == ContourPrimitive.RECTANGLE:
            nr_points_a = int(curve_primitive.a * curve_primitive.res)
            nr_points_b = int(curve_primitive.b * curve_primitive.res)
            a = np.linspace(-0.5*curve_primitive.a, 0.5*curve_primitive.a, nr_points_a)
            b = np.linspace(-0.5 * curve_primitive.b, 0.5 * curve_primitive.b, nr_points_b)
            h = np.linspace(curve_primitive.h_start, curve_primitive.h_end, 2*(nr_points_a+nr_points_b))

            # rectangle starts in top left corner, center is in the middle
            curve_points = np.zeros(shape=(2*(nr_points_a+nr_points_b), 3))
            curve_points[0:nr_points_a] = np.array(
                [a, nr_points_a*[curve_primitive.b * 0.5], h[0:nr_points_a]]).transpose()
            curve_points[nr_points_a:nr_points_a+nr_points_b] = np.array(
                [nr_points_b*[curve_primitive.a * 0.5], -b, h[nr_points_a:nr_points_a+nr_points_b]]).transpose()
            curve_points[nr_points_a + nr_points_b:2 * nr_points_a + nr_points_b] = np.array(
                [-1.0 * a, nr_points_a*[curve_primitive.b * (-0.5)], h[nr_points_a + nr_points_b:2 * nr_points_a + nr_points_b]]).transpose()
            curve_points[2 * nr_points_a + nr_points_b:] = np.array(
                [[-0.5 * curve_primitive.a] * nr_points_b, b, h[2 * nr_points_a + nr_points_b:]]).transpose()

            if curve_primitive.ccw:
                curve_points = np.flip(curve_points, axis=0)

            curve_normals = np.array([curve_primitive.normal for _ in range(2*(nr_points_a+nr_points_b))])

        # transform points and normals
        rot = tftrans.quaternion_matrix([curve_primitive.pose.orientation.x,
                                         curve_primitive.pose.orientation.y,
                                         curve_primitive.pose.orientation.z,
                                         curve_primitive.pose.orientation.w])
        trans = tftrans.translation_matrix([curve_primitive.pose.position.x,
                                            curve_primitive.pose.position.y,
                                            curve_primitive.pose.position.z])

        curve_points = np.dot(np.hstack([curve_points, np.ones((curve_points.shape[0], 1))]), rot)
        curve_points = np.dot(curve_points, trans)
        curve_normals = np.dot(curve_normals, rot[:3, :3])

        # make sure, the velocity is set to some reasonable range
        # exec_options.cart_vel_limit = max(min(exec_options.cart_vel_limit, 0.2), 0.005)

        local_trans = tftrans.translation_matrix([cp.pose.position.x,
                                                 cp.pose.position.y,
                                                 cp.pose.position.z])
        local_rot = tftrans.quaternion_matrix([cp.pose.orientation.x,
                                              cp.pose.orientation.y,
                                              cp.pose.orientation.z,
                                              cp.pose.orientation.w])
        curve_shifted = np.dot(local_rot, curve_points.transpose()).transpose()
        curve_shifted = np.dot(local_trans, curve_shifted.transpose()).transpose()
        curve_points = curve_shifted

        # remove duplicate points from data
        to_delete = np.where(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1) <= 0.0001)
        curve_points = np.delete(curve_points, to_delete, axis=0)
        curve_normals = np.delete(curve_normals, to_delete, axis=0)

        # # applying
        self.custom_contour = curve_points[:, :2]
        rospy.loginfo('Pattern is {}'.format(pattern))
        if not (pattern is None or pattern == "None"):
            rospy.loginfo('Applying pattern {}'.format(pattern))
            self.applyCurve()
            # magic to add z to the soup
            # self.curve_pcd = o3d.geometry.PointCloud()

            # interpolate the height value of points
            x = np.linspace(0.0, 1.0, curve_points.shape[0])
            f_points = interpolate.interp1d(x[:], curve_points.transpose())

            curve_points = np.hstack((self.curve.squeeze(), f_points(np.linspace(0,1,self.curve.shape[0])).transpose()[:,2] [:, np.newaxis]))

            x = np.linspace(0.0, 1.0, curve_normals.shape[0])
            f_normal = interpolate.interp1d(x[:], curve_normals.transpose())
            curve_normals = f_normal(np.linspace(0,1,self.curve.shape[0])).transpose()

        # curve_points = np.hstack((self.curve.squeeze(), np.repeat(curve_points[:, 2], self.CURVE_UPSAMPLING_FACTOR)[:, np.newaxis]))
        # curve_normals = np.repeat(curve_normals, self.CURVE_UPSAMPLING_FACTOR, axis=0)

        self.last_execution_result = False
        self.is_done = False
        goal = CurveExecutionGoal(
                header=Header(frame_id='table2'),
                opt=exec_options,
                curve=[Point(**{k: v for k, v in zip(["x", "y", "z"], p)}) for p in np.array(curve_points)],
                normals=[Vector3(**{k: v for k, v in zip(["x", "y", "z"], n)}) for n in np.array(curve_normals)],
            )

        LoggerProxy.logger(action=LoggingTask.RESET)
        LoggerProxy.logger(action=LoggingTask.SETREFERENCE, data=goal)
        LoggerProxy.logger(action=LoggingTask.START)

        def my_done_cb(success_code, msg):
            LoggerProxy.logger(action=LoggingTask.STOP)
            if success_code == actionlib.GoalStatus.SUCCEEDED:
                fp = os.path.expanduser('~/traj_complete_log/') + (log_name or 'log_file') + ".json"
                LoggerProxy.logger(action=LoggingTask.SAVE, data=fp)
                print('Log saved to {}'.format(fp))
            else:
                print('Execution failed, not saving the log!')
            self.done_cb(success_code, msg)

        self.action_client.send_goal(
            goal,
            my_done_cb, self.active_cb, self.feedback_cb
        )

        while not self.is_done:
            self.rate.sleep()
        rospy.signal_shutdown("I am done!")

    def run(self):
        """This is the main loop function. It first creates the relevant OpenCV window and control components.
        Afterwards, it loops over, redraws the image and listens for user inputs.
        """
        cv2.namedWindow("contour image")
        cv2.createTrackbar('contour scale', self.window_name, 128, 256, self.adjust_contours)
        cv2.createTrackbar('offset X', self.window_name, 128, 256, self.adjust_offsetX)
        cv2.createTrackbar('offset Y', self.window_name, 128, 256, self.adjust_offsetY)
        cv2.createTrackbar('num of reps', self.window_name, self.num_pattern_reps, 128, self.set_numOfReps)
        cv2.createTrackbar('pattern type', self.window_name, np.where(np.array(self.patternChoices) == self.patternType)[0], len(self.patternChoices) - 1, self.change_curveType)
        cv2.createTrackbar('approx method', self.window_name, int(np.log2(self.approx_method)), 2, self.change_approxMethod)
        cv2.createTrackbar('z computation', self.window_name, int(self.z_comp == "flat"), 1, self.change_z_computation)
        cv2.createTrackbar('pattern rotation', self.window_name, 180 + self.pattern_rotation, 360, self.change_pattern_rotation)
        cv2.createTrackbar('pattern gap', self.window_name, 128 + self.pattern_gap, 256, self.change_pattern_gap)
        cv2.createTrackbar('exec engine', self.window_name, self.exec_engine, 3, self.change_exec_engine)
        cv2.createTrackbar('cart speed (cm/s)', self.window_name, int(self.cart_speed * 100), 20, self.change_cart_speed)
        cv2.createTrackbar('smooth curve', self.window_name, int(self.c_smoothing_enabled), 1, self.change_c_smooth)
        cv2.createTrackbar('trim start', self.window_name, self.pattern_trim_start, 99, self.change_trim_start)
        cv2.createTrackbar('trim trail', self.window_name, self.pattern_trim_trail, 99, self.change_trim_trail)
        cv2.createTrackbar('pcl interpolation', self.window_name, int(self.pcl_interpolation), 1, self.change_pcl_interp)
        cv2.createTrackbar('show orig pcl', self.window_name, int(self.show_orig_pcl), 1, self.change_show_orig_pcl)
        cv2.createTrackbar('progress', self.window_name, 0, 100, lambda x: cv2.setTrackbarPos("bla" if self.is_executing else "progress", "contour image", 0))
        cv2.setMouseCallback("contour image", self.mouse_cb)
        key = 0

        while not rospy.is_shutdown():
            if not cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE):  # user closed window
                rospy.signal_shutdown("User closed the window.")
                return

            self.info_panel.fill(0)  # reset info panel
            # redraw infos
            ofst = 18
            cv2.putText(self.info_panel, self.patternType, (10, ofst), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (64, 255, 0), 2)
            cv2.putText(self.info_panel, "Z-axis: {}".format(self.z_comp), (240, ofst), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 255, 0), 1)
            cv2.putText(self.info_panel, "approx method: {}".format(ApproxMethod.toString(self.approx_method)), (500, ofst), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 255, 0), 1)
            cv2.putText(self.info_panel, "exec eng: {}{}".format(self._convert_exec_engine(self.exec_engine)[0],  " ({} m/s)".format(self.cart_speed) if self.exec_engine == 1 else ""), (10, int(ofst * 2.2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (64, 255, 0), 1)

            cv2.imshow("contour image", np.vstack((cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB), self.info_panel)))
            key = cv2.waitKey(20)

            if key & 0xFF == ord("q"):  # quit
                rospy.signal_shutdown("User requested shutdown via GUI.")
                return
            elif key & 0xFF == ord(" "):  # pause message updates
                self.pauseUpdate = not self.pauseUpdate
            elif key & 0xFF == ord("x"):  # pause message updates
                if self.is_executing:
                    self.action_client.cancel_goal()
                    # self.action_client.stop_tracking_goal()
            elif key & 0xFF == ord("c"):  # delete custom contours
                self.custom_contour = None
                self.redrawContours()
            elif key & 0xFF == ord("s"):  # cycle select contour
                self.cycleSelectContour()
            elif key & 0xFF == ord("a"):  # draw curve
                if not self.is_drawing:
                    self.applyCurve()
                else:
                    rospy.logwarn("Currently drawing a custom contour, cannot apply curve!")
            elif key & 0xFF == 13:  # compute normals, curve and send curve
                if self.curve is not None and not self.is_executing:
                    if self.compute3DCurve():
                        self.sendCurve()
                elif self.is_executing:
                    rospy.logwarn("Curve execution in progress, cannot send another one!")
                else:
                    rospy.logwarn("There is no curve generated, cannot send to controller.")
            elif key != -1:
                rospy.logdebug("Unknown key pressed: {}".format(key))

            self.rate.sleep()

    def perception_cb(self, color_msg, contours_msg, pcl_msg=None):
        """Message callback. Gathers color image, contours and pcl messages and applies basic processing.
        """
        rospy.logwarn("update!")
        if not self.pauseUpdate:  # self.pauseUpdate == user requested that the image is not updated
            try:
                # get tf from depth -> rgb to align the PCL
                self.depth_to_color_tf = self.tf_buffer.lookup_transform(self.color_frame_id, self.pcl_frame_id, pcl_msg.header.stamp)
                self.color_to_table_tf = self.tf_buffer.lookup_transform(self.table_frame_id, self.color_frame_id, pcl_msg.header.stamp)
                # self.depth_to_color_tf = self.tf_buffer.lookup_transform(self.pcl_frame_id, self.color_frame_id, pcl_msg.header.stamp)
                # self.color_to_table_tf = self.tf_buffer.lookup_transform(self.color_frame_id, self.table_frame_id, pcl_msg.header.stamp)
            except Exception as e:
                rospy.logerr("Could not get depth_to_color or color_to_table transform because: {}".format(e))
            else:
                self.input_contours = contour_msg2list(contours_msg)
                self.input_image = self._bridge.imgmsg_to_cv2(color_msg, "passthrough").copy()
                self.selected_contour = -1  # reset selected contour, as the c. order no longer applies
                self.recalcContours()
                self.xyz, self.rgb = ftl_pcl2numpy(pcl_msg)
                self.last_update = color_msg.header.stamp

    def recalcContours(self, autoRedraw=True):
        """Recomputes the current contours. Namely, applies rescaling per user request.

        Args:
            autoRedraw (bool, optional): Whether to also update the image in GUI at the end. Defaults to True.
        """
        # create empty mask
        mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
        # paint contours as areas into it
        for c in self.input_contours:
            mask = cv2.fillPoly(mask, [c], (255, 255, 255), cv2.LINE_AA)
        # just some post-processing to remove noise & stuff
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernelClose)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelOpen)
        # contour scaling
        if self.contour_scale > 0:
            mask = cv2.dilate(mask, self.kernelDilate, iterations=self.contour_scale)
        elif self.contour_scale < 0:
            mask = cv2.erode(mask, self.kernelDilate, iterations=-self.contour_scale)
        # recompute contours
        self.contours, c_hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        offset = np.r_[self.contour_offsetX, self.contour_offsetY]
        self.contours = [c + offset for c in self.contours]
        # redraw contour image, if necessary
        if autoRedraw:
            self.redrawContours()

    def redrawContours(self):
        """Updates the image in GUI. Draws the current contours into the color image.
        """
        self.contour_image = cv2.drawContours(self.input_image.copy(), self.contours, -1, (0, 128, 128), 1)  # draw contours into image
        if not self.contours:
            return
        if self.custom_contour is not None and len(self.custom_contour) > 0:  # draw custom contours, if any
            cv2.drawContours(self.contour_image, [np.array(self.custom_contour).reshape(-1, 1, 2)], 0, (255, 0, 0), 1)
        elif self.selected_contour >= 0 and self.selected_contour < len(self.contours):  # draw selected contour, if any
            cv2.drawContours(self.contour_image, self.contours, self.selected_contour, (0, 0, 255), 1)
        if self.curve is not None:  # draw generated curve, if it exists
            curve = np.round(self.curve).astype(np.int32)
            cv2.drawContours(self.contour_image, [curve], 0, (0, 255, 0), 1)

    def destroy(self):
        # called when exiting
        if self.is_executing:
            self.action_client.cancel_goal()
            self.action_client.stop_tracking_goal()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node('interface_node')

    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", "-n", action="store_true", default=False, help="A flag to start the node without the user interface (default params are read from yaml, mode of operation  must be *single*).")
    parser.add_argument("--test1", "-t", action="store_true", default=False, help="A flag to start the node without the user interface and just send a test curve to the control node.")
    parser.add_argument("--show-pcl", "-p", action="store_true", default=False, help="Turns on PCL visualization.")
    parser.add_argument("--full-pcl", "-f", action="store_true", default=False, help="Draws 3D curve on top of the entire PCL.")
    parser.add_argument("--original-pcl", "-o", action="store_true", default=False, help="Draws 3D curve on top of the entire PCL.")
    parser.add_argument("--save-curve", "-s", action="store_true", default=False, help="same as nogui but instead of sending to action server it stores curve as npz.")
    parser.add_argument("--config", "-c", nargs=1, type=str, default="config/default_configuration.yaml", help="Path to configuration file, relative to the package.")
    parser.add_argument("--experiment", "-e", nargs="+", default=[], help="")
    parser.add_argument("--artificial", "-a", type=str, default='', help="Compose and send artificial curve from a dictionary.")
    args = parser.parse_args()

    rospack = rospkg.RosPack()
    with open(os.path.join(rospack.get_path('traj_complete_ros'), args.config), "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print('config: {}'.format(cfg))

    if (args.nogui or args.save_curve) and not args.test1:
        ui = UI(config=cfg, show_pcl=args.show_pcl, full_pcl=args.full_pcl, original_pcl=args.original_pcl)
        ui.selectAndSendCurve(save=args.save_curve)
    elif args.test1:
        cfg['pattern_rotation'] = 20.0
        cfg['num_of_reps'] = 10
        ui = UI(config=cfg, no_wait=True, show_pcl=args.show_pcl, full_pcl=args.full_pcl)
        cp = ContourPrimitive()
        cp.a = 0.2
        cp.b = 0.1
        cp.ccw = False
        cp.pose.position.x = 0.2
        cp.pose.orientation.w = 1
        cp.type = ContourPrimitive.SPIRAL
        cp.r_start = 0.05
        cp.r_end = 0.05
        cp.rounds = 1.0
        cp.normal = [0, 0, 1]
        cp.h_start = 0.02
        cp.h_end = 0.02
        cp.res = 100
        opt = ExecutionOptions()
        # opt.executor_engine = ExecutionOptions.MOVEIT_CART_PLANNING
        opt.executor_engine = ExecutionOptions.DESCARTES_TOLERANCED_PLANNING
        opt.cart_vel_limit = 0.1
        # opt.executor_engine = ExecutionOptions.RELAXEDIK
        opt.tool_orientation = ExecutionOptions.USE_FIXED_ORIENTATION
        # ui.SendTestCurve(curve_primitive=cp, exec_options=opt, pattern='hand_knot_sfr')
        ui.SendTestCurve(curve_primitive=cp, exec_options=opt, pattern='None')

    elif args.artificial:
        ui = UI(config=cfg, no_wait=True, show_pcl=args.show_pcl, full_pcl=args.full_pcl)
        ui.color_to_table_tf = TransformStamped(
            transform=Transform(
                translation=Vector3(0.2, 0, 0.0),
                rotation=Quaternion(0, 0, 0, 1)
            )
        )
        try:
            ui.composeArtificialCurve(args.artificial)
            ui.sendCurve()
        except Exception as e:
            print(e)
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            sys.exit(13)

    elif args.experiment:
        data = yaml.safe_load(args.experiment[0])
        print(args.experiment[0])
        rospy.loginfo("Running experiment {}".format(data["name"]))

        # set config params
        fields = ["num_of_reps", "default_pattern", "approx_method", "pcl_interpolation", "cart_vel_limit", "executor_engine", "c_smoothing_enabled", "tool_orientation", "pattern_rotation", "pattern_trim_start", "pattern_trim_trail"]
        for f in fields:
            if f not in data:
                continue
            v = data[f]
            if f in cfg and v is not None and ((type(v) is str and bool(v)) or (type(v) is float and not np.isnan(v))):
                cfg[f] = v

        if "real" in data["data_type"].lower():  # test on real data
            ui = UI(config=cfg, no_wait=False, show_pcl=args.show_pcl, full_pcl=args.full_pcl)
            try:
                ui.selectAndSendCurve(log_name=data["name"] + "_" + data["last_time"])
            except Exception as e:
                print(e)
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                sys.exit(13)
        else:
            # cfg['pattern_rotation'] = data['pattern_rotation']
            print('config')
            cfg['num_of_reps'] = 10
            print(cfg)
            ui = UI(config=cfg, no_wait=True, show_pcl=args.show_pcl, full_pcl=args.full_pcl)
            cp = ContourPrimitive()
            cp.a = data["a"] if not np.isnan(data["a"]) else 0.2
            cp.b = data["b"] if not np.isnan(data["b"]) else 0.1
            cp.ccw = bool(data["ccw"])
            cp.rounds = data["rounds"] if not np.isnan(data["rounds"]) else 1.0
            cp.pose.position.x = data["x"] if not np.isnan(data["x"]) else 0.2
            cp.pose.position.y = data["y"] if not np.isnan(data["y"]) else 0.0
            cp.pose.position.z = data["z"] if not np.isnan(data["z"]) else 0.0
            cp.pose.orientation.w = 1
            cp.type = data["cp_type"]
            cp.normal = [0, 0, 1]
            cp.r_start = data["r_start"] if not np.isnan(data["r_start"]) else 0.3
            cp.r_end = data["r_end"] if not np.isnan(data["r_end"]) else 0.3
            cp.h_start = data["h_start"] if not np.isnan(data["h_start"]) else 0.3
            cp.h_end = data["h_end"] if not np.isnan(data["h_end"]) else 0.3
            cp.res = data["res"] if not np.isnan(data["res"]) else 100
            pattern = data["default_pattern"]
            # print([getattr(cp, att) for att in dir(cp) if not att.startswith("_")])
            opt = ExecutionOptions()
            _, opt.executor_engine = UI._convert_exec_engine(data["executor_engine"])
            opt.cart_vel_limit = data["cart_vel_limit"] if not np.isnan(data["cart_vel_limit"]) else 0.1
            opt.tool_orientation = data["tool_orientation"] if not np.isnan(data["tool_orientation"]) else 11
            try:
                ui.SendTestCurve(curve_primitive=cp, exec_options=opt, log_name=str(data["name"]) + "_" + data["last_time"], pattern=pattern)
            except Exception as e:
                print(e)
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                sys.exit(13)
        if ui.last_execution_result:
            sys.exit(0)
        else:
            sys.exit(13)
    else:
        ui = UI(config=cfg, show_pcl=args.show_pcl, full_pcl=args.full_pcl)
        ui.run()

    rospy.loginfo('exiting')
