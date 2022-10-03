# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract images and depth data from a rosbag.
"""
from __future__ import print_function, division
import os
from typing import Optional
# import cv2
from tf_bag import BagTfTransformer
import os
import rosbag
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import tf2_ros
from std_msgs.msg import String
from tf.transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import tf
from numpy import genfromtxt, math
from sensor_msgs.msg import CameraInfo
from scipy.spatial import ConvexHull, Delaunay
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import copy
from scipy.spatial.transform import Rotation
from rospkg import RosPack
from traj_complete_ros.utils import BsplineGen
from traj_complete_ros.plotting import rotation_bw2_vectors, plot3D, plot_quaternions, plot_quivers, generate_orientations, plot_curve_o3d
from scipy.optimize import curve_fit


class myBagTransformer(object):
    def __init__(self, path, bagName):
        self.bag = rosbag.Bag(os.path.join(path, bagName), "r")
        self.bag_transformer = BagTfTransformer(self.bag)

    def lookupTransform(self, orig_frame, dest_frame, t):
        # orig_frame = 'tracker_LHR_786752BF'
        # dest_frame = 'world'

        trans, rot = self.bag_transformer.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame,
                                                          time=t)
        return trans, rot

    def tracker_to_end(self, tracker_topic):
        # provides transform between world_vive to board
        trans = np.zeros([1, 3], dtype=float)
        rot = np.zeros([1, 4], dtype=float)
        count = 0
        for topic, msg, t in self.bag.read_messages(topics=['/tf_static']):
            if count != 1:
                if msg.transforms[0].header.frame_id == tracker_topic:
                    if msg.transforms[0].child_frame_id == "end_point":
                        print(msg)
                        trans = np.array([msg.transforms[0].transform.translation.x,
                                          msg.transforms[0].transform.translation.y,
                                          msg.transforms[0].transform.translation.z])
                        rot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                        msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                        count = 1
            else:
                break

        return trans, rot

    def world_to_board(self):
        # provides transform between world_vive to board
        trans = np.empty([1, 3], dtype=float)
        rot = np.empty([1, 4], dtype=float)
        count = 0
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            if count != 1:
                if msg.transforms[0].header.frame_id == "world_vive":
                    if msg.transforms[0].child_frame_id == "board":
                        print(msg)
                        trans = np.array([msg.transforms[0].transform.translation.x,
                                          msg.transforms[0].transform.translation.y,
                                          msg.transforms[0].transform.translation.z])
                        rot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                        msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                        count = 1
            else:
                break
        return trans, rot

    # topic "camera1_color_optical_frame", "camera2_color_optical_frame"
    def world_to_camera(self, topicC):
        # provides transform between world_vive to camera (using world_to_board transform)
        Ctran = np.empty([1, 3], dtype=float)
        Crot = np.empty([1, 4], dtype=float)
        found = False
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            if msg.transforms[0].header.frame_id == "board":
                if msg.transforms[0].child_frame_id == topicC:
                    print(msg)
                    Ctran = np.array([msg.transforms[0].transform.translation.x,
                                      msg.transforms[0].transform.translation.y,
                                      msg.transforms[0].transform.translation.z])
                    Crot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                     msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                    found = True
                    break

        if not found:
            return

        # getting transform from world_vive to board
        Btran, Brot = self.world_to_board()
        Rt_world_to_board = self.make_Rt_matrix(Btran, Brot)
        # Rt_world_to_board = quaternion_matrix(Brot)
        # Rt_world_to_board[0:3,3] = Btran

        # getting transform from board to camera
        Rt_board_to_camera = self.make_Rt_matrix(Ctran, Crot)
        # Rt_board_to_camera = quaternion_matrix(Crot)
        # Rt_board_to_camera[0:3,3] = Ctran

        # getting transform from world_vive to camera
        Rt_world_to_camera = np.matmul(Rt_world_to_board, Rt_board_to_camera)
        return Rt_world_to_camera

    # topic "camera1_color_optical_frame", "camera2_color_optical_frame"
    def transform_from_to(self, from_topic, to_topic):
        # provides tranform between from_topic parent frame to to_topic (child_frame) - only between neigboring nodes
        count1 = 0
        trans = np.empty([1, 3], dtype=float)
        rot = np.empty([1, 4], dtype=float)
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            if count1 != 1:
                if msg.transforms[0].header.frame_id == from_topic:
                    if msg.transforms[0].child_frame_id == to_topic:
                        # print(msg)
                        trans = np.array([msg.transforms[0].transform.translation.x,
                                          msg.transforms[0].transform.translation.y,
                                          msg.transforms[0].transform.translation.z])
                        rot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                        msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                        count1 = 1
            else:
                break
        return trans, rot

    def make_Rt_matrix(self, trans, rot):
        # gives back rotation_translation matrix 4x4 from translation and rotation vector
        Rt_mat = quaternion_matrix(rot)
        Rt_mat[0:3, 3] = trans
        return Rt_mat


def normalize_data(data, add_angle=0):
    # data[..., :2] -= (np.min(data, axis=0)[0], data[0, 1])  # normalize on mim/min
    data[..., :2] -= data[0, :2]  # normalize: first point -> [0, 0]
    diff = data[-1, :2] - data[0, :2]
    diff /= np.linalg.norm(diff)
    a = np.arctan(diff[1] / diff[0]) + np.deg2rad(add_angle)
    dcm = Rotation.from_euler("z", -a, degrees=False).as_matrix()
    dcm = np.pad(dcm, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    dcm[-1, -1] = 1
    return np.dot(dcm, np.pad(data, ((0, 0), (0, 1)), mode="constant", constant_values=1).T).T[..., :3]


def filter_close_points(data, orients, threshold=5e-4):
    n = data.shape[0]
    good_indices = np.linalg.norm(data[:-1, :] - data[1:, :], axis=1) > threshold
    data = np.vstack((data[0, :], data[1:, :][good_indices, :]))
    orients = np.vstack((orients[0, :], orients[1:, :][good_indices, :]))
    if data.shape[0] < n:
        data, orients = filter_close_points(data, orients, threshold)
    return data, orients


def project_point(point, RT_mat, K_mat):
    point_in_cam = np.dot(np.linalg.inv(RT_mat), point)
    point_camera = np.dot(K_mat, point_in_cam[:3])
    point_camera /= point_camera[2]
    point_image = tuple(point_camera[:2].astype(int))
    return point_image, point_camera


def main():
    suffix = ""
    smoothing_enabled = True
    close_point_filtering = True
    resampling_enabled = True
    lifting_enabled = False
    saving_enabled = True
    bspline_n_points = 1024
    # library_name = "hand_trajectories.yaml"
    library_name = "default_configuration.yaml"

    draw_steps = False
    show_image_projection = False
    save_image_projection = False
    bagNames = [
        # ("patterns/2021-02-22-11-38-41.bag", "test1")
        # ("patterns/old/bag_2021-01-29-17-30-16.bag", "zigzag"),
        # ("patterns/old/bag_2021-01-29-17-30-48.bag", "s-wave"),
        # ("patterns/old/bag_2021-01-29-17-31-00.bag", "z-wave"),
        # ("patterns/old/bag_2021-01-29-17-31-17.bag", "hand_knot"),
        # ("patterns/old/bag_2021-01-29-17-31-54.bag", "triangle")

        # ("patterns/bag_2021-02-23-15-00-17.bag", ""),
        # ("patterns/bag_2021-02-23-15-00-32.bag", "Z-wave"),
        ("patterns/bag_2021-02-23-15-00-49.bag", "smooth_knot"),
        # ("patterns/bag_2021-02-23-15-01-31.bag", "knot_circle_1"),
        # ("patterns/bag_2021-02-23-15-02-23.bag", "knot_circle_2"),
        # ("patterns/bag_2021-02-23-15-03-00.bag", "knot_circle_3"),
        # ("patterns/bag_2021-02-23-15-03-24.bag", "knot_circle_4"),
        # ("patterns/bag_2021-02-23-15-03-55.bag", "circle_1"),
        # ("patterns/bag_2021-02-23-15-04-18.bag", "circle_1"),
        # ("patterns/bag_2021-02-23-15-05-11.bag", "knot_straight"),
        # ("patterns/bag_2021-02-23-15-05-28.bag", "S-wave")
    ]

    t_diff_threshold = rospy.Duration.from_sec(0.02)
    key_t_diff_threshold = rospy.Duration.from_sec(0.5)
    smoothing_kernel_size = 11
    smoothing_sigma = 5.0
    smoothing_convolution = "valid"
    lift_threshold = 3e-4
    lift_preset_height = 0.5e-2

    rp = RosPack()
    if saving_enabled:
        library_path = os.path.join(rp.get_path("traj_complete_ros"), "config", library_name)

    for bagName, patternName in bagNames:

        path, bagName = os.path.split(bagName)

        bagfName = ""

        bag = rosbag.Bag(os.path.join(path, bagName), "r")

        # bag_transformer = BagTfTransformer(bag)

        mbt = myBagTransformer(path, bagName)
        # RtWorldToCam = mbt.world_to_camera('camera2_color_optical_frame')
        RtWorldToCam = mbt.world_to_camera('camera1_color_optical_frame')
        if RtWorldToCam is None:
            print("Camera 1 images not found in the bag.")
            RtWorldToCam = mbt.world_to_camera('camera2_color_optical_frame')
            if RtWorldToCam is None:
                raise Exception("Could not find any cameras in the bag")
            else:
                print('RtWorldToCam2', RtWorldToCam)
                image_topic = '/camera2/color'
        else:
            print('RtWorldToCam1', RtWorldToCam)
            image_topic = '/camera1/color'

        for topic, msg, t in bag.read_messages('{}/camera_info_throttle'.format(image_topic)):
            K_matrix = np.array(msg.K).reshape((3, 3))
            break
        # print(K_matrix)

        image_topic = next((name for name, info in bag.get_type_and_topic_info(
        ).topics.items() if image_topic in name and info.msg_type == "sensor_msgs/CompressedImage"))
        bridge = CvBridge()
        count = 0
        output_folder = os.path.join(path, bagName[0:-4])

        tracker_topic = next((name for name, info in bag.get_type_and_topic_info().topics.items() if "tracker" in name))
        # tracker_topic = "/tracker_LHR_1D894210"  # or manually

        tracker_msgs = list(bag.read_messages(tracker_topic))
        # tracker_time = np.array([rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs) for topic, msg, t in tracker_msgs])

        common_time = mbt.bag_transformer.waitForTransform("world_vive", "board")
        vive2world_translation, vive2world_quaternion = mbt.bag_transformer.lookupTransform(
                        "world_vive", "board", common_time)
        # RtBoard2Vive = mbt.make_Rt_matrix(np.array(vive2world_translation), np.array(vive2world_quaternion))
        # RtBoard2Vive = np.linalg.inv(mbt.make_Rt_matrix(np.array(vive2world_translation), np.array(vive2world_quaternion)))
        RtBoard2Vive = mbt.make_Rt_matrix(np.array(vive2world_translation), np.array(vive2world_quaternion))

        key_msgs_time = np.array([msg.timestamp for msg in bag.read_messages("/key") if msg.message.data == "q"])
        last_key_pressed = None
        traj_started = False

        end_point, _ = mbt.tracker_to_end(tracker_topic)

        img_msgs = list(bag.read_messages(image_topic))
        # img_time = np.array([msg.timestamp for msg in img_msgs])
        img_time = np.array([msg.message.header.stamp for msg in img_msgs])

        print("Found {} timestamps in the bag.".format(len(tracker_msgs)))

        data = []
        tracker_data = []
        tracker_data_pixel = []
        orient = []
        raw_points = []
        raw_quats = []
        data_2d = []
        data_pixel = []
        for topic, msg, t in tracker_msgs:
            # refTime = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)  # TODO: We should use the timestamp of the pose, not the message!!!
            refTime = copy.deepcopy(msg.header.stamp)
            messageIndex = np.argmin(np.abs(img_time - refTime))
            diff = abs(img_time[messageIndex] - refTime)

            key_time = key_msgs_time[abs(key_msgs_time - refTime).argmin()]
            key_diff = abs(key_time - refTime)
            if key_diff < t_diff_threshold:
                if last_key_pressed is None or abs(key_time - last_key_pressed) > key_t_diff_threshold:
                    last_key_pressed = key_time
                    traj_started ^= True

            if not traj_started:
                continue

            tracker_pose = msg.pose
            world_to_tracker2_translation, world_to_tracker2_quaternion = tracker_pose.position, tracker_pose.orientation
            trans, rot = np.array([getattr(world_to_tracker2_translation, v) for v in "xyz"]), np.array([getattr(world_to_tracker2_quaternion, v) for v in "xyzw"]) # TODO: why change to wxyz from xyzW?
            raw_points.append(trans)
            raw_quats.append(rot)
            RtWorldToTracker = mbt.make_Rt_matrix(trans, rot)

            tip_pos = np.dot(RtWorldToTracker, np.hstack((end_point, 1)).T)
            pose = np.dot(RtWorldToTracker, np.hstack((end_point, 0)).T)
            data.append(tip_pos)
            orient.append(pose)

            point_image, point_camera = project_point(tip_pos, RtWorldToCam, K_matrix)
            data_2d.append(point_camera)  # 2D coordinates (float precision)
            data_pixel.append(point_image)  # 2D coordinates (int/pixel precision)
            tracker_data.append(trans)
            point_image, _ = project_point(np.hstack((trans, 1)).T, RtWorldToCam, K_matrix)
            tracker_data_pixel.append(point_image)

            # if True:
            if diff < t_diff_threshold:
                # continue
                # image data saving
                img = bridge.compressed_imgmsg_to_cv2(img_msgs[messageIndex].message, desired_encoding="passthrough")
                fname = bagfName + 'C1' + "F%04i.png" % count
                # cv2.imwrite(os.path.join(output_folder + "/Image/", fname1), cv_image1)

                frame = "F%04i" % count

                for pt, tpt in zip(data_pixel, tracker_data_pixel):
                    img = cv2.line(img, tpt, pt, (255, 128, 128), thickness=1, lineType=cv2.LINE_AA)
                    cv2.drawMarker(img, pt, (0, 0, 255), markerType=cv2.MARKER_DIAMOND,
                                   markerSize=2, thickness=1, line_type=cv2.LINE_AA)

                if show_image_projection:
                    cv2.imshow("image", img)
                    cv2.waitKey(10)

                if save_image_projection:
                    fname_output = os.path.join(output_folder, 'image_{:04d}.png'.format(messageIndex))
                    directory = os.path.split(fname_output)[0]
                    if not os.path.isdir(directory):
                        os.mkdir(directory)
                    cv2.imwrite(fname_output, img)

                print("Wrote image %i" % count)
            else:
                print("Skipped image %i" % count)

            count += 1
        bag.close()

        data = np.array(data)[..., :3]
        orient = np.array(orient)[..., :3]
        # orient /= np.linalg.norm(orient, axis=1)[..., np.newaxis]
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            plot3D(ax, data, color="black")
            plt.show(block=False)
            # plot3D(fig.add_subplot(121, projection='3d'), data)

        raw2d, filtered2d, smooth2d = None, None, None
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        if draw_steps:
            raw2d = normalize_data(np.pad(PCA(n_components=2).fit_transform(copy.deepcopy(data)), ((0, 0), (0, 1)), mode="constant", constant_values=0))
        if close_point_filtering:
            # remove very close points
            data, orient = filter_close_points(data, orient)
            if draw_steps:
                filtered2d = normalize_data(np.pad(PCA(n_components=2).fit_transform(copy.deepcopy(data)), ((0, 0), (0, 1)), mode="constant", constant_values=0))
        if smoothing_enabled:  # smoothing
            half = smoothing_kernel_size >> 1

            curve_3d_padded = np.vstack((
                np.repeat(data[:1, :], half, axis=0),
                data[..., :],
                np.repeat(data[-1:, :], half, axis=0)
            ))
            gkernel = np.tile(signal.gaussian(smoothing_kernel_size, smoothing_sigma)[:, np.newaxis], (1, 3))
            curve_3d_smooth = signal.fftconvolve(curve_3d_padded, gkernel, mode=smoothing_convolution, axes=0)
            ret, M, _ = cv2.estimateAffine3D(curve_3d_smooth.astype(np.float32), data.astype(np.float32))
            M = np.vstack((M, np.r_[0, 0, 0, 1]))
            if ret:
                data = np.dot(M, cv2.convertPointsToHomogeneous(curve_3d_smooth).squeeze().T).T

        data2_pca = PCA(n_components=2)
        data_2 = data2_pca.fit_transform(data[:, :3])
        data_2 = np.pad(data_2, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        data3_pca = PCA(n_components=3)
        # normalize data
        data_2 = normalize_data(data_2, 0)
        if draw_steps:
            smooth2d = copy.deepcopy(data_2)

        # >>> PLOTING THE ORIENTATIONS <<<
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # raw_points = normalize_data(np.array(raw_points))
        # raw_quats = np.array(raw_quats)
        # raw_orients = np.array([Rotation.from_quat(r).as_euler("xyz", degrees=False) for r in raw_quats])
        # raw_orients = np.array([Rotation.from_quat(r).as_rotvec() for r in raw_quats])
        # raw_orients = np.array([Rotation.from_quat(r).apply([1, 0, 0]) for r in raw_quats])
        # d2 = raw_points
        # d2 = raw_points[:, :2]
        # plot3D(ax, d2 - raw_orients, autoscale=True, color="red")
        # ax.quiver3D(*np.hstack((d2, d2 - raw_orients)).T, length=0.1, normalize=True, arrow_length_ratio=0.03, pivot="tip", linewidths=0.4)
        # plot3D(ax, d2, autoscale=True, color="green")

        # ax.quiver3D(*np.hstack((data_2, data_2 - orient)).T, length=0.1, normalize=True, arrow_length_ratio=0.1, pivot="tip", linewidths=0.4)

        # orient = np.sin(np.array([(i / d_len) for i in range(d_len)]) * np.pi * 4)

        # ax.quiver3D(*np.hstack((data_2, data_2 - orient)).T, length=0.1, normalize=True, arrow_length_ratio=0.1, pivot="tip", linewidths=0.4)
        # plot3D(ax, data_2, autoscale=True, color="green")
        # ax.quiver3D(*np.hstack((data[:, :3], data[:, :3] - orient * 0.9)).T, length=0.1, normalize=True, arrow_length_ratio=0.03, pivot="tip", linewidths=0.4)
        # plot3D(ax, data, autoscale=True, color="green")

        # ax.set_xlim3d([-0.5, 0])
        # ax.set_ylim3d([-1, -0.5])
        # ax.set_zlim3d([-2, 0])
        # plt.show(block=True)
        # orients, rot_vecs = generate_orientations(data_2, center_point=data_2[0, :])
        # quats = np.array([Rotation.from_rotvec(rv).as_quat() for rv in rot_vecs])
        # plot_quivers(data_2, orients)


        if resampling_enabled:
            padding_amount = 12
            up_samp_amount, down_samp_amount = 2, 4
            data_resamp = signal.resample_poly(np.pad(data_2, ((padding_amount, padding_amount), (0, 0)), mode="edge"), up_samp_amount, down_samp_amount, axis=0, window=("kaiser", 10))
            _, uidx = np.unique(data_resamp, return_index=True, axis=0)
            unpad_amount = int(padding_amount * (up_samp_amount / down_samp_amount))
            data_resamp = data_resamp[np.sort(uidx), :][unpad_amount:-unpad_amount, :]
            # plot3D(ax, data_resamp, autoscale=True, color="red")
            data_2 = data_resamp

        # generate artificial orientations
        orients, rotations = generate_orientations(data_2, angle=np.deg2rad(5))   # the pattern will have this orientation
        quats = np.array([r.as_quat() for r in rotations])

        # 2D PATTERN
        # wayPoints = data_2[..., :2]
        # bsgen = BsplineGen.fromWaypoints(wayPoints)

        # (2 + 1 + 4)D PATTERN (pos + z + quaternion)
        wayPoints = np.hstack((data_2[..., :2], np.zeros((data_2.shape[0], 1)), quats))
        bsgen = BsplineGen.fromWaypoints7D(wayPoints.T)

        # orients, rotations = generate_orientations(data_2, center_point=data_2[0, :] * 2)
        # plot_quivers(data_2, orients)

        # try to generate new data from the bspline
        sampled_spline_points = bsgen.generate_bspline_sample(500)
        points = sampled_spline_points[:, :3]
        quats = sampled_spline_points[:, 3:]
        # plot_quaternions(points, quats)
        plot_curve_o3d(points, quats)
        npz_file = os.path.join("quat_curves", f"{patternName}.npz")
        np.savez(npz_file, points=data_2, orientations=quats)

        if draw_steps:
            # raw
            ff, a = plt.subplots()
            a.set_title("raw")
            a.plot(*raw2d[:, :2].T)
            a.scatter(*raw2d[:, :2].T, c="red", marker="x")
            # raw zoomed
            ff, a = plt.subplots()
            a.set_title("raw")
            a.plot(*raw2d[:, :2].T)
            a.scatter(*raw2d[:, :2].T, c="red", marker="x")
            a.set_xlim(-0.001, 0.002)
            a.set_ylim(-0.003, 0.003)

            if filtered2d is not None:
                ff, a = plt.subplots()
                a.set_title("filtered")
                filtered2d[:, 0] *= -1
                a.plot(*filtered2d[:, :2].T)
                a.scatter(*filtered2d[:, :2].T, c="red", marker="x")
                a.set_xlim(-0.001, 0.002)
                a.set_ylim(-0.003, 0.003)
            if smooth2d is not None:
                ff, a = plt.subplots()
                a.set_title("smooth")
                smooth2d[:, 0] *= -1
                a.plot(*smooth2d[:, :2].T)
                a.scatter(*smooth2d[:, :2].T, c="red", marker="x")
            ff, a = plt.subplots()
            tmp = copy.deepcopy(data_2)
            tmp[:, 0] *= -1
            a.plot(*tmp[:, :2].T)
            a.scatter(*tmp[:, :2].T, c="red", marker="x")
            a.set_title("final (resamp)")
            plt.show()

        if saving_enabled:
            tmp_suff = "_" + suffix if suffix else ""
            bsgen.appendToLibrary(library_path, name=patternName + tmp_suff)

        if lifting_enabled:
            data_lift = copy.deepcopy(data_2)
            is_up = (data[..., 2] - data_2[..., 2]) > lift_threshold
            data_lift[is_up, 2] += lift_preset_height
            # plot3D(ax, data_lift, autoscale=True, color="green")

        plt.show()


if __name__ == '__main__':
    main()
