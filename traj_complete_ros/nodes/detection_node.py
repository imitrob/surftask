#!/usr/bin/env python
import rospy
import message_filters
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from traject_msgs.msg import ContourArrayStamped
import numpy as np
from scipy.interpolate import griddata
import open3d as o3d
from traj_complete_ros.utils import contours2msg, ftl_pcl2numpy
from traj_complete_ros.plane_model import PlaneModel
from traj_complete_ros.cfg import DetectionConfig
from dynamic_reconfigure.server import Server
from time import time


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


class MicroTimer():

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.startTime = time()
        self.lastLapTime = self.startTime

    def lap(self, section="", ending=""):
        if not self.enabled:
            return
        self.lapTime = time()
        elapsedLap = self.lapTime - self.lastLapTime
        elapsedTolal = self.lapTime - self.startTime
        if self.lastLapTime == self.startTime:
            preamb = "==========>\n"
        else:
            preamb = ""
        self.lastLapTime = self.lapTime
        rospy.loginfo("{}Elapsed time:\n\tper section {} = {}\n\ttotal = {}{}".format(preamb, "({})".format(section) if section else "", elapsedLap, elapsedTolal, ending))

    def end(self, section=""):
        self.lap(section=section, ending="\n<==========")


class Detector(object):
    PHASE_PLANE_CALIBRATION = 0
    PHASE_DETECTION = 1
    USE_PCD_PROCESSING = False
    PUBLISH_CONTOUR_IMAGE = False

    def __init__(self):
        # self._camera_pose = PoseStamped()
        depth_topic = rospy.get_param("/depth_topic", '/camera/aligned_depth_to_color/image_raw')
        depth_camInfo_topic = rospy.get_param("/depth_topic", '/camera/aligned_depth_to_color/camera_info')
        color_topic = rospy.get_param("/color_topic", '/camera/color/image_raw')
        pcl_topic = rospy.get_param("/pcl_topic", '/camera/depth/color/points')
        contour_topic = rospy.get_param("/contour_topic", '/contour/raw')
        plane_max_dist = rospy.get_param("~plane_max_dist", 0.01)  # 1 cm
        self.n_plane_calibration_samples = rospy.get_param("~n_plane_calibration_samples", 1e5)  # 1e6 @848x480 => equals to 3 frames

        self._bridge = CvBridge()

        try:
            depth_msg = rospy.wait_for_message(depth_topic, Image, 15.0)
            depth_camInfo_msg = rospy.wait_for_message(depth_camInfo_topic, CameraInfo, 15.0)
            rospy.wait_for_message(color_topic, Image, 15.0)
            self._depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=5)
            self._color_sub = message_filters.Subscriber(color_topic, Image, queue_size=5)
            if self.USE_PCD_PROCESSING:  # PCD processing is insanely slow!
                self._pcl_sub = message_filters.Subscriber(pcl_topic, PointCloud2, queue_size=5)
                sync_topics = [self._color_sub, self._depth_sub, self._pcl_sub]
            else:
                sync_topics = [self._color_sub, self._depth_sub]

            self._image_time_sync = message_filters.ApproximateTimeSynchronizer(sync_topics, 5, slop=0.05)
        except rospy.ROSException as e:
            print(e)
            exit(22)
        else:
            self.contour_pub = rospy.Publisher(contour_topic, ContourArrayStamped, queue_size=10)
            self._image_pub = rospy.Publisher("/contour/image", Image, queue_size=10)

            rpos, cpos = np.mgrid[0:depth_msg.height, 0:depth_msg.width]
            self.depth_grid = (rpos, cpos)
            self.cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_msg.width, depth_msg.height, *np.r_[depth_camInfo_msg.K][[0, 4, 2, 5]])
            self.camera_matrix = np.reshape(depth_camInfo_msg.K, (3, 3))
            self.distCoeffs = np.array(depth_camInfo_msg.D)

            self.plane_model = PlaneModel(plane_max_dist, ransac_n=30, iterations=1000)
            self.phase = self.PHASE_PLANE_CALIBRATION

            self.min_contour_len = 200
            self.blur_size = 11
            self.blur_sigma = 5
            self.morph_kernel = 3
            self.bilateral_filter_size = 9
            self.bilateral_sigma_color = 30
            self.bilateral_sigma_space = 30
            self.auto_canny_sigma = 0.7
            self.use_graph_cut = False
            self.contour_approximation_delta = 0.5
            self.use_convex_fill_poly = True
            self.use_convex_hull = False
            self.mask_closing_iterations = 5
            self.mask_opening_iterations = 3
            self.mask_kernel = 5
            self.cfg_first_run = True
            self.srv = Server(DetectionConfig, self.config_cb)
            self.vis_enabled = False
            self._image_time_sync.registerCallback(self.perception_cb)

    def config_cb(self, config, level):
        ret_msg = ""
        for k, v in config.items():
            if hasattr(self, k):
                # default_v = config["groups"]["parameters"][k]
                old_v = getattr(self, k)
                if old_v != v:
                    if k == "blur_size":
                        if v % 2 != 1:
                            v += 1
                    config[k] = v
                    setattr(self, k, v)
                    ret_msg += "Parameter {} changed: {} -> {}\n".format(k, old_v, v)
            else:
                if k == "groups":
                    continue
                rospy.logwarn("Detector object does not have attribute specified in dynamic reconfigure request: {}.".format(k))
        rospy.loginfo(ret_msg)
        self.cfg_first_run = False
        return config

    def makeMaskFromContours(self, contours):
        contour_mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
        for c in contours:
            if self.use_convex_fill_poly:
                contour_mask = cv2.fillConvexPoly(contour_mask, c, 255, cv2.LINE_8)
            else:
                contour_mask = cv2.fillPoly(contour_mask, c, 255, cv2.LINE_8)
        return contour_mask

    def perception_cb(self, color, depth, pcl_msg=None):
        assert isinstance(color, Image)
        assert isinstance(depth, Image)
        timer = MicroTimer(enabled=False)
        color_img = self._bridge.imgmsg_to_cv2(color, "passthrough").copy()

        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HLS)

        if self.phase == self.PHASE_PLANE_CALIBRATION:
            if pcl_msg is None:
                self.phase = self.PHASE_DETECTION
            else:
                depth_img = self._bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough").copy()
                samples = np.where(np.logical_and(~np.isnan(depth_img), depth_img > 0))  # take all pixels that are not NaN and have value greater than zero
                depth_interp = np.array(griddata(np.array(samples).T, depth_img[samples], self.depth_grid, method="nearest"), dtype=np.uint16).astype(np.uint16)
                timer.lap("interpolation")
                depth_img = o3d.geometry.Image(depth_interp)
                pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, self.cam_intrinsics)
                self.plane_model.addSamplesOpen3d(pcd)
                if self.plane_model.n_samples > self.n_plane_calibration_samples:
                    rospy.loginfo("Gathered enough samples, computing plane...")
                    self.plane_model.compute()
                    self.phase = self.PHASE_DETECTION
                    rospy.loginfo("Plane model computed, switching to detection phase.")
                else:
                    rospy.loginfo("Gathered {} samples out of {} ({}%).".format(self.plane_model.n_samples, self.n_plane_calibration_samples, self.plane_model.n_samples / self.n_plane_calibration_samples * 100))

        if self.phase == self.PHASE_DETECTION:
            if pcl_msg is not None:
                xyz, rgb = ftl_pcl2numpy(pcl_msg)
                filtering_idx = self.plane_model.argFilter(xyz)  # filter plane
                xyz_filtered = xyz[filtering_idx]
                # rgb_as_f = rgb.astype(np.float) / 255  # normalize to <0, 1> range
                # rgb_filtered = rgb[filtering_idx]
                timer.lap("pcl extraction")

                # contruct a PCL in order to cluster the points (could be sped up in the future by just using plane DBSCAN)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_filtered.astype(np.float64))
                timer.lap("pcd creation")
                labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=200, print_progress=False))
                unq_labels = np.unique(labels, return_counts=True)
                # order_cluster_labels = unq_labels[0][np.argsort(unq_labels[1])]
                largest_clusters = np.isin(labels, unq_labels[0][np.where(unq_labels[1] > unq_labels[1].max() * 0.1)][1:])  # select largest clusters
                timer.lap("pcd clustering")

                zilch = np.zeros((3, 1), dtype=np.float32)
                points_2d, _ = cv2.projectPoints(xyz_filtered[largest_clusters], zilch, zilch, self.camera_matrix, self.distCoeffs)
                points_2d = np.round(points_2d.squeeze()).astype(np.int16)
                timer.lap("pcd projection")

                # Mask
                mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.float32)
                mask[points_2d[..., 1], points_2d[..., 0]] = 1
                kernelDilate = np.ones((7, 7), np.uint8)
                mask = cv2.dilate(mask, kernelDilate, iterations=2)
                mask = cv2.GaussianBlur(mask, (self.blur_size, self.blur_size), self.blur_sigma)
                timer.lap("mask processing")

            # filtering
            filtered = cv2.bilateralFilter(color_img, self.bilateral_filter_size, self.bilateral_sigma_color, self.bilateral_sigma_space)
            if self.vis_enabled:
                cv2.imshow("Bilateral filter", filtered)
            img = auto_canny(filtered, sigma=self.auto_canny_sigma)
            if self.vis_enabled:
                cv2.imshow("Canny", img)
            if pcl_msg is not None:
                img = (img * mask).astype(np.uint8)

            # masking contour manipulation
            contours, c_hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            maskKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self.mask_kernel, self.mask_kernel))
            if self.vis_enabled:
                tmp = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
                cv2.imshow("Canny contours", tmp)
            if self.use_graph_cut:
                contour_mask = self.makeMaskFromContours(contours)
                gc_mask = np.zeros(color_img.shape[:2], np.uint8) + 2
                gc_mask[contour_mask == 255] = 1
                contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_DILATE, maskKernel, iterations=20)
                gc_mask[contour_mask == 0] = 0
                if self.vis_enabled:
                    tmp = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
                    cv2.imshow("GC mask", (gc_mask * 127).astype(np.uint8))
                gc_bgdModel = np.zeros((1, 65), np.float64)
                gc_fgdModel = np.zeros((1, 65), np.float64)
                cv2.grabCut(color_img, gc_mask, None, gc_bgdModel, gc_fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                contour_mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
                contour_mask[(gc_mask == 1) | (gc_mask == 3)] = 255
                contours, c_hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if self.vis_enabled:
                    tmp = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
                    cv2.imshow("GC contours", tmp)

            c_lens = [cv2.arcLength(c, True) for c in contours]
            contours = [cv2.approxPolyDP(c, self.contour_approximation_delta * c_len, closed=True) for c, c_len in zip(contours, c_lens) if c_len > self.min_contour_len]
            if self.vis_enabled:
                tmp = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
                cv2.imshow("approximated contours", tmp)
            timer.lap("contour pre-processing")

            # draw detected areas to an empty image
            contour_mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
            cv2.drawContours(contour_mask, contours, -1, 255, 1)
            contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, maskKernel, iterations=self.mask_closing_iterations)
            if self.vis_enabled:
                cv2.imshow("Contour mask", contour_mask)

            contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour_mask = self.makeMaskFromContours(contours)
            contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, maskKernel, iterations=self.mask_opening_iterations)
            timer.lap("contour post-processing")

            contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cimg = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
            # cv2.imshow("cmask", cimg)
            # cv2.waitKey(20)
            if self.use_convex_hull:
                contours = [cv2.convexHull(c) for c in contours]
                if self.vis_enabled:
                    tmp = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
                    cv2.imshow("Convex hull contours", tmp)
            timer.lap("final contour extraction")

            if self.PUBLISH_CONTOUR_IMAGE:
                img = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1, cv2.LINE_AA)
                self._image_pub.publish(self._bridge.cv2_to_imgmsg(img, encoding="rgb8"))

            if self.vis_enabled:
                # c = contours[0]
                # mask = np.zeros((self.cam_intrinsics.height, self.cam_intrinsics.width), dtype=np.uint8)
                # mask = cv2.fillPoly(mask, [c], (255, 255, 255), cv2.LINE_AA)
                # c, c_hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # tmp = cv2.drawContours(color_img.copy(), c, -1, (0, 255, 0), 1)
                tmp = cv2.drawContours(color_img.copy(), contours, -1, (0, 255, 0), 1)
                cv2.imshow("Final contours", tmp)
                cv2.waitKey(0)
            # contour
            if contours is not None and len(contours) > 0:
                cont_msg = contours2msg(contours, depth.header.stamp, depth.header.frame_id)
                self.contour_pub.publish(cont_msg)
            timer.end("publishing")


if __name__ == "__main__":
    rospy.init_node('detection_node')

    rospy.sleep(1.0)
    det = Detector()
    rospy.spin()

    rospy.loginfo('exiting')
