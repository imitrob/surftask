import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Union


def plot3D(ax, data, marker=".", color="blue", autoscale=True):
    if data.shape[1] < data.shape[0]:
        data = data.T
    if data.shape[0] == 4:
        xs, ys, zs, _ = data
    elif data.shape[0] == 3:
        xs, ys, zs = data
    else:
        raise Exception("Wrong data shape?")

    ax.plot(xs, ys, zs, marker=marker, color=color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if autoscale:
        ax.margins(0, 0, 0)
        all_data = np.zeros((3, 0))
        for c in ax.get_children():
            if hasattr(c, "get_data_3d"):
                # print(np.shape(np.array(c.get_data_3d())))
                cdata = np.array(c.get_data_3d())
                all_data = np.hstack((cdata, all_data))

        dsize = np.max(all_data.max(axis=1) - all_data.min(axis=1)) * 0.8
        dcenter = all_data.mean(axis=1)
        ax.set_xlim(dcenter[0] - dsize, dcenter[0] + dsize)
        ax.set_ylim(dcenter[1] - dsize, dcenter[1] + dsize)
        ax.set_zlim(dcenter[2] - dsize, dcenter[2] + dsize)


def rotation_bw2_vectors(a, b):
    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rvec = np.cross(a, b)
    rvec = (rvec / np.linalg.norm(rvec)) * angle
    r = Rotation.from_rotvec(rvec)
    return r


def generate_quaternions(points: np.ndarray, center_point: Optional[np.ndarray] = None, angle: Optional[float] = None, default_tool_orientation: np.ndarray = np.r_[0, 0, -1]):
    # d_len = points.shape[0]
    assert points.shape[1] == 2, "Points must have shape [n, 2], where n is the number of points."
    if center_point is None:
        center_point = points.mean(axis=0)
    # rot_vecs = [np.cross(p - center_point, z_axis) for p in points]
    dir_vecs = np.array([p - center_point for p in points])
    if angle is not None:
        dir_vecs = dir_vecs / np.linalg.norm(dir_vecs, axis=1)[:, np.newaxis]
        z_axis = np.r_[0, 0, angle]
    else:
        z_axis = np.r_[0, 0, 1]

    x_axis = np.r_[1, 0, 0]
    x2tool_rotation = rotation_bw2_vectors(x_axis, default_tool_orientation)
    # x2tool_rotation = rotation_bw2_vectors(default_tool_orientation, x_axis)

    rot_vecs = [np.cross(dv, z_axis) for dv in dir_vecs]
    rotations = [Rotation.from_rotvec(rv) * x2tool_rotation for rv in rot_vecs]

    quaternions = [r.as_quat() for r in rotations]
    return quaternions


def generate_orientations(points: np.ndarray, center_point: Optional[np.ndarray] = None, angle: Optional[float] = None, default_tool_orientation: np.ndarray = np.r_[0, 0, -1]):
    if center_point is None:
        center_point = points.mean(axis=0)
    dir_vecs = points - center_point
    if angle is not None:
        dir_vecs = dir_vecs / np.linalg.norm(dir_vecs, axis=1)[:, np.newaxis]
        z_axis = np.r_[0, 0, angle]
    else:
        z_axis = np.r_[0, 0, 1]
    x_axis = np.r_[1, 0, 0]
    # x2tool_rotation = rotation_bw2_vectors(x_axis, default_tool_orientation)
    x2tool_rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

    rot_vecs = [np.cross(dv, z_axis) for dv in dir_vecs]
    rotations = [Rotation.from_rotvec(rv) * x2tool_rotation for rv in rot_vecs]
    orientations = np.array([r.apply(x_axis) for r in rotations])
    return orientations, rotations


def plot_curve_o3d(points, quaternions, default_orientation:np.ndarray = np.r_[0, 0, -1], scale=None, axis_scale=None, show_axis=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Curve visualization", width=1920, height=1012)

    default_orientation = default_orientation.astype(float)
    if scale is None:
        scale = np.linalg.norm(points.mean(axis=0))
    if axis_scale is None:
        axis_scale = np.linalg.norm(points.mean(axis=0))
    default_orientation *= scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0.0, 0.2])
    vis.add_geometry(pcd)

    orientation_vectors = np.array([Rotation.from_quat(q).apply(default_orientation) for q in quaternions]) + points
    # orientation_vectors = np.array([Rotation.from_euler("xyz", (0, np.pi / 4, 0)).apply(default_orientation) for i in range(points.shape[0])]) + points

    pcd_orient = o3d.geometry.PointCloud()
    pcd_orient.points = o3d.utility.Vector3dVector(orientation_vectors)
    # pcd.paint_uniform_color([0.0, 0.0, 0.6])
    # vis.add_geometry(pcd_orient)

    if show_axis:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_scale, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)

    pcd_lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, pcd_orient, [(i, i) for i in range(points.shape[0])])
    pcd.paint_uniform_color([0.0, 0.5, 0.1])
    vis.add_geometry(pcd_lines)

    vis.run()
    vis.capture_screen_image("curve_plot.png")
    vis.destroy_window()


def plot_quivers(points, orientations, arrow_length=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # plot3D(ax, orientations, autoscale=True, color="blue")
    ax.quiver3D(*np.hstack((points, orientations + points)).T, normalize=False, arrow_length_ratio=0.1, pivot="tip", linewidths=0.4)
    # ax.quiver3D(*np.hstack((points, orientations + points)).T, length=arrow_length, normalize=True, arrow_length_ratio=0.1, pivot="tip", linewidths=0.4)
    # ax.quiver3D(*np.hstack((points, orientations - points)).T, length=0.1, normalize=True, arrow_length_ratio=0.1, pivot="tip", linewidths=0.4)
    ax.view_init(elev=10., azim=10)
    plot3D(ax, points, autoscale=True, color="red")
    plt.show(block=False)


def plot_quaternions(points, quaternions, arrow_length=0.1):
    x_axis = np.r_[1, 0, 0] * arrow_length
    # tool_default_orientation = np.r_[0, 0, -1]
    orientations = np.array([Rotation.from_quat(q).apply(x_axis) for q in quaternions])
    plot_quivers(points, orientations, arrow_length)


def quat2vec(quaternions):
    z_axis = np.r_[0, 0, 1]
    return np.array([Rotation.from_quat(quat).apply(z_axis) for quat in quaternions])
