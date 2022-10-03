import collections
import numpy as np
from scipy import signal, sparse, interpolate
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import cv2
from cv_bridge import CvBridge

import rosbag
import rospy
import sensor_msgs.point_cloud2
from traj_complete_ros.utils import BsplineGen
from traj_complete_ros.utils import smooth_contours_w_butterworth, set_up_2dplot, cont_pt_by_pt, plot_displacement_scatter
from scipy import stats
from traj_complete_ros.plotting import plot_quivers, quat2vec, plot_quaternions, plot_curve_o3d


class ApproxMethod:
    TAN = 1
    REG = 2
    ZERO = 4

    @classmethod
    def fromString(cls, string):
        try:
            return next(v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(getattr(cls, k)) and string.lower() in k.lower())
        except StopIteration:
            return 4

    @classmethod
    def toString(cls, val):
        try:
            return next(k for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(getattr(cls, k)) and v == val)
        except StopIteration:
            return "unknown"


def apply_pattern_to_contour(pattern_bsplineP, traj_bspline, no_of_prds, use_gap=0, pattern_rotation=0, pattern_trim_start=0, pattern_trim_trail=0, approx_method=ApproxMethod.TAN, numer_of_points=92, debug=True):
    """ Shift bspline indices so that patterns dont end on a corner if possible

    Args:
        pattern_bsplineP (BsplineGen): spline defining the selected pattern to apply (taken from library or elsewhere), ideally 1 repetition
        traj_bspline (Nx2 array): sampled spline defining the contour along which we want to apply the pattern
        no_of_prds (int): number of repetition to apply the pattern along contour - pattern will be adequatelly scaled to fit along the contour in the given amount of repetitions
        use_gap (int, optional): Number of points to be used as a gap between pattern repetitions. Defaults to 0.
        pattern_rotation (int, optional): Rotation for the pattern in degrees
        pattern_trim (int, optional): point trimming of the pattern in percent
        approx_method (ApproxMethod, optional): Approximation method to be used. Defaults to ApproxMethod.TAN.

    Returns:
        [type]: traj_bspline with pattern superimposed on top of it.
    """

    # sample pattern
    #TODO based on the number of repetitions resample traj_spline so it is nicely dividable
    # traj_bspline=resample_spline(traj_bspline, 1000)
    pattern_bsplineP.nmbPts = numer_of_points
    pattern_bspline_s = pattern_bsplineP.generate_bspline_sample()
    # plot_displacement_scatter(trnsfrmd_waypts, 'transfrmd waypts')
    nmbPts_patternPlus = round(traj_bspline.shape[0] / no_of_prds)
    nmbPts_patternOnly = (traj_bspline.shape[0] / no_of_prds) - 2*use_gap

    # pattern rotation
    if pattern_rotation != 0:
        pattern_bspline_s = rotate_vector7D(pattern_bspline_s,pattern_rotation)

    if pattern_bspline_s[0,  0] > pattern_bspline_s[-1,  0]:
        pattern_bspline_s = np.flipud(pattern_bspline_s)

    # pattern trimming
    n = pattern_bspline_s.shape[0]
    if pattern_trim_start > 0:
        idx = int(n * (float(pattern_trim_start) / 100))
        if idx > 0:
            pattern_bspline_s = pattern_bspline_s[idx:, :]
    if pattern_trim_trail > 0:
        idx = int(n * (float(pattern_trim_trail) / 100))
        if idx > 0:
            pattern_bspline_s = pattern_bspline_s[:-idx, :]

    # pattern_bspline = generate_bspline_sample(pattern_bsplineP.nmbPts,pattern_bsplineP.pars,pattern_bsplineP.xout)
    # traj_bsplineS = traj_bspline

    # TODO: why are moved last M values before the first n-M values? Where M = n/number of repentitions (no_of_prds)
    traj_bspline = np.vstack(
        (traj_bspline[int(traj_bspline.shape[0] / no_of_prds):-1, :],
         traj_bspline[0:int(traj_bspline.shape[0] / no_of_prds), :]))

    # approximate bspline using endpoints of graphical input bspline
    if approx_method:
        shftd_apprx_bspline_s = compute_apprx_spline_pattern(pattern_bspline_s, approx_method, nmbPts_patternPlus)

    # resample pattern_bspline
    pattern_bspline_s = resample_spline(pattern_bspline_s, nmbPts_patternPlus)
    if debug:
        plot_displacement_scatter(pattern_bspline_s[:,:2], 'transfrmd waypts fltd bspline')

    # TODO: check if the changes below are working to make desired repetitions and gaps

    # for manual implementation of constrained transformation using pseudoinverse
    M = np.zeros((shftd_apprx_bspline_s.shape[0] * 2, 4))
    for n in range(0, M.shape[0], 2):
        # assert np.round(n/2) == n/2
        n_half = int(n // 2)
        M[n, :] = np.hstack((shftd_apprx_bspline_s[n_half, :], np.ones(1), np.zeros(1)))
        M[n + 1, :] = np.hstack(
            (shftd_apprx_bspline_s[n_half, -1], -1.0 * shftd_apprx_bspline_s[n_half, 0], np.zeros(1), np.ones(1)))

    trnsfrmd_apprx_bspline = np.zeros(traj_bspline.shape)
    trnsfrmd_waypts = np.zeros(
        (traj_bspline.shape[0], pattern_bspline_s.shape[1]))
    # # TODO: check
    # j_range = range(0, trnsfrmd_apprx_bspline.shape[0], shftd_apprx_bspline.shape[0] + 2*use_gap)
    # k_range = range(0, trnsfrmd_waypts.shape[0], traj_bspline.shape[0]/no_of_prds)

    # beginning of the pattern
    # j_range = range(use_gap, trnsfrmd_apprx_bspline.shape[0], shftd_apprx_bspline_s.shape[0] + 2*use_gap)
    j_range = range(0, trnsfrmd_apprx_bspline.shape[0], shftd_apprx_bspline_s.shape[0])
    # end of the pattern
    k_range = range(0, trnsfrmd_waypts.shape[0], nmbPts_patternPlus)
    for j, k in zip(j_range, k_range):

        A_r1 = np.hstack((np.cos(0), -np.sin(0), np.zeros(1)))
        A_r2 = np.hstack((np.sin(0), np.cos(0), np.zeros(1)))
        A_r1r2 = np.vstack((A_r1, A_r2))
        A_r3 = np.hstack((np.zeros(2), np.ones(1)))
        mean_approxPattern = np.mean(shftd_apprx_bspline_s,axis=0)
        mean_trajectory = np.mean(traj_bspline[j:j + shftd_apprx_bspline_s.shape[0], :],axis =0)
        trans = mean_trajectory - mean_approxPattern
        moved_approxPattern = shftd_apprx_bspline_s + trans
        moved_approxPattern3d = np.concatenate((moved_approxPattern, np.zeros((moved_approxPattern.shape[0], 1))), axis=1)
        moved_traj3D = np.concatenate((traj_bspline[j:j + shftd_apprx_bspline_s.shape[0], :], np.zeros((traj_bspline[j:j + shftd_apprx_bspline_s.shape[0], :].shape[0], 1))), axis=1)
        if moved_approxPattern3d.shape != moved_traj3D.shape:
            print('catch the error. Set your debug point on this line and analyze what happened.')
        estRot = R.align_vectors(moved_approxPattern3d, moved_traj3D)
        theta = np.degrees(estRot[1])

        if int(cv2.__version__[0]) < 4:
            A_r1r2 = cv2.estimateRigidTransform(shftd_apprx_bspline_s.reshape(1, -1, 2),
                                            traj_bspline[j:j + shftd_apprx_bspline_s.shape[0], :].reshape(1, -1, 2),
                                            fullAffine=False)
        else:
            A_r1r2, _ = cv2.estimateAffinePartial2D(shftd_apprx_bspline_s.reshape(1, -1, 2),
                                            traj_bspline[j:j + shftd_apprx_bspline_s.shape[0], :].reshape(1, -1, 2))
        if A_r1r2 is None:
            print('Failed to estimate rigid transformation, trying manual implementation')
            A_elems = np.dot(np.linalg.pinv(M), traj_bspline[j:j + shftd_apprx_bspline_s.shape[0], :].reshape(-1, 1))

            # shear, coordinate flipping and non-uniform scaling constrained
            A_r1 = A_elems[0:3].reshape(-1)
            A_r2 = np.hstack((-1.0 * A_elems[1], A_elems[0], A_elems[3]))
            A_r1r2 = np.vstack((A_r1, A_r2))

        A_mat = np.vstack((A_r1r2, A_r3))
        trnsfrmd_apprx_bspline[j:j + shftd_apprx_bspline_s.shape[0], :] = cv2.transform(shftd_apprx_bspline_s[:,:2].reshape(1, -1, 2), A_mat[0:2])[0]
        #transform the pattern to the baseline
        trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], :2] = cv2.transform(pattern_bspline_s[:,:2].reshape(1, -1, 2), A_mat[0:2])[0]
        #copy z values from pattern
        trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], 2] = pattern_bspline_s[:,2].copy()
        #rotate the quaternion around the z axis by the angle theta
        # theta = np.degrees(np.arccos(A_mat[0, 0]))
        points3D = rotate_vector7D(pattern_bspline_s, theta, plot_flag = False)
        trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], 3:] = points3D[:,3:].copy()
        if debug:
            # plot_curve_o3d(points3D[:,:3],points3D[:,3:])
            # plot_curve_o3d(trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], :3],trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], 3:])
            plot_quaternions(trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], :3],points3D[:,3:])

            # plot_displacement_scatter(trnsfrmd_waypts[:,:2], 'approx bspline')

            # plot_quivers(trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], :3], quat2vec(trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], 3:]))  # plots the data before rotation
            # plot_quaternions(trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], :3], trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], 3:])  # plots the data after rotating quaternions
            # plot_quaternions(trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], :3],
            #              trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], 3:])  # plots the data after rotating quaternions
    # for i in range(0, len(j_range)-2):
    #     if use_gap > 0:
    #         # create connection bspline sampled by use_gap nmb of points
    #         connect_bspline = BsplineGen()
    #         #connect_xy = np.array([[trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-2, 0], trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-2, 1]], [trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-1, 0],
    #         #                                                                                                                                                                          trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-1, 1]], [trnsfrmd_apprx_bspline[j_range[i+1], 0], trnsfrmd_apprx_bspline[j_range[i+1], 1]], [trnsfrmd_apprx_bspline[j_range[i+1]+1, 0], trnsfrmd_apprx_bspline[j_range[i+1]+1, 1]]])
    #         connect_xy = np.array([[trnsfrmd_apprx_bspline[j_range[i+1] - use_gap-1, 0], trnsfrmd_apprx_bspline[j_range[i+1] -use_gap-1, 1]], [trnsfrmd_apprx_bspline[j_range[i+1] -use_gap, 0],
    #                   trnsfrmd_apprx_bspline[j_range[i] - use_gap, 1]], [trnsfrmd_apprx_bspline[j_range[i+1]+ use_gap, 0], trnsfrmd_apprx_bspline[j_range[i+1]+use_gap, 1]],
    #                                [trnsfrmd_apprx_bspline[j_range[i+1]+use_gap+1, 0], trnsfrmd_apprx_bspline[j_range[i+1]+use_gap+1, 1]]])
    #
    #         connect_bspline.generate_bspline_pars(connect_xy, per_flag=False)
    #         connect_bspline.nmbPts = 2*use_gap
    #         connect_bspline_sampled = connect_bspline.generate_bspline_sample()
    #         trnsfrmd_apprx_bspline[j_range[i+1]-use_gap:j_range[i+1]+use_gap,0] = connect_bspline_sampled[:,0]
    #         trnsfrmd_apprx_bspline[j_range[i+1]-use_gap:j_range[i+1]+use_gap,1] = connect_bspline_sampled[:,1]
    #         print('')
    #         #TODO add to trnsfrmd_apprx_bspline j_range[i]+ shftd_apprx_bspline.shape[0]: j_range[i+1] - 1
        ##############################################
    if use_gap > 0:
        for i in range(0, len(k_range) - 1):
            # create connection bspline sampled by use_gap nmb of points
            connect_bspline = BsplineGen()
            # connect_xy = np.array([[trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-2, 0], trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-2, 1]], [trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-1, 0],
            #                                                                                                                                                                          trnsfrmd_apprx_bspline[j_range[i] + shftd_apprx_bspline.shape[0]-1, 1]], [trnsfrmd_apprx_bspline[j_range[i+1], 0], trnsfrmd_apprx_bspline[j_range[i+1], 1]], [trnsfrmd_apprx_bspline[j_range[i+1]+1, 0], trnsfrmd_apprx_bspline[j_range[i+1]+1, 1]]])
            connect_xy = np.array([[trnsfrmd_waypts[k_range[i + 1] - use_gap - 1, 0],
                                    trnsfrmd_waypts[k_range[i + 1] - use_gap - 1, 1]],
                                   [trnsfrmd_waypts[k_range[i + 1] - use_gap, 0],
                                    trnsfrmd_waypts[k_range[i + 1] - use_gap, 1]],
                                   [trnsfrmd_waypts[k_range[i + 1] + use_gap, 0],
                                    trnsfrmd_waypts[k_range[i + 1] + use_gap, 1]],
                                   [trnsfrmd_waypts[k_range[i + 1] + use_gap + 1, 0],
                                    trnsfrmd_waypts[k_range[i + 1] + use_gap + 1, 1]]])

            connect_bspline.generate_bspline_pars(connect_xy, per_flag=False)
            connect_bspline.nmbPts = 2 * use_gap
            connect_bspline_sampled = connect_bspline.generate_bspline_sample()
            trnsfrmd_waypts[k_range[i + 1] - use_gap:k_range[i + 1] + use_gap, 0] = connect_bspline_sampled[:, 0]
            trnsfrmd_waypts[k_range[i + 1] - use_gap:k_range[i + 1] + use_gap, 1] = connect_bspline_sampled[:, 1]
        connect_xy = np.array([[trnsfrmd_waypts[k_range[len(k_range)-1] + pattern_bspline_s.shape[0] - use_gap - 1, 0],
                                    trnsfrmd_waypts[k_range[len(k_range)-1] + pattern_bspline_s.shape[0] - use_gap - 1, 1]],
                                   [trnsfrmd_waypts[k_range[len(k_range)-1] + pattern_bspline_s.shape[0]- use_gap, 0],
                                    trnsfrmd_waypts[k_range[len(k_range)-1] + pattern_bspline_s.shape[0]- use_gap, 1]],
                                   [trnsfrmd_waypts[use_gap, 0],
                                    trnsfrmd_waypts[use_gap, 1]],
                                   [trnsfrmd_waypts[use_gap + 1, 0],
                                    trnsfrmd_waypts[use_gap + 1, 1]]])
        connect_bspline.generate_bspline_pars(connect_xy, per_flag=False)
        connect_bspline.nmbPts = 2 * use_gap
        connect_bspline_sampled = connect_bspline.generate_bspline_sample()
        trnsfrmd_waypts[k_range[len(k_range)-1] + pattern_bspline_s.shape[0] - use_gap:k_range[len(k_range)-1] + pattern_bspline_s.shape[0], 0] = connect_bspline_sampled[0:use_gap, 0]
        trnsfrmd_waypts[k_range[len(k_range)-1] + pattern_bspline_s.shape[0]- use_gap:k_range[len(k_range)-1] + pattern_bspline_s.shape[0], 1] = connect_bspline_sampled[0:use_gap, 1]
        trnsfrmd_waypts[0:use_gap, 0] = connect_bspline_sampled[use_gap:2*use_gap, 0]
        trnsfrmd_waypts[0:use_gap, 1] = connect_bspline_sampled[use_gap:2*use_gap, 1]

        # TODO add to trnsfrmd_apprx_bspline j_range[i]+ shftd_apprx_bspline.shape[0]: j_range[i+1] - 1
    if debug:
        plot_curve_o3d(trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], :3],
                     trnsfrmd_waypts[:k + pattern_bspline_s.shape[0], 3:])  # plots the data after rotating quaternions

    # smooth countour for shape matching,
    # WARNING: too low cutoff frequency causes start and end of contour to disconnect
  #  trnsfrmd_waypts_sgnl_fltd = smooth_contours_w_butterworth(trnsfrmd_apprx_bspline, 5, 0.4, 'low')
  #   plot_displacement_scatter(trnsfrmd_waypts, 'transfrmd waypts')
    trnsfrmd_waypts_sgnl_fltd = smooth_contours_w_butterworth(trnsfrmd_waypts[:,:2], 5, 0.4, 'low')
    #trnsfrmd_waypts_fltd = np.array(zip(list(trnsfrmd_waypts_sgnl_fltd.real), list(trnsfrmd_waypts_sgnl_fltd.imag)))
    # python 3 - zip needs to be packed in one more list around...
    trnsfrmd_waypts_fltd = np.array(list(zip(list(trnsfrmd_waypts_sgnl_fltd.real), list(trnsfrmd_waypts_sgnl_fltd.imag))))
    trnsfrmd_waypts_fltd = np.concatenate((trnsfrmd_waypts_fltd,trnsfrmd_waypts[:,2:]),axis = 1)

    applied_bspline = BsplineGen()
    applied_bspline.generate_bspline_pars(trnsfrmd_waypts_fltd, per_flag=True)
    applied_bspline.generate_bspline_pars7D(np.transpose(trnsfrmd_waypts_fltd), per_flag=True)
    applied_bspline.nmbPts = 2 * traj_bspline.shape[0]
    applied_bspline_s = applied_bspline.generate_bspline_sample()
    # applied_bspline = generate_bspline(2 * traj_bspline.shape[0], trnsfrmd_waypts_fltd, per_flag=True)

    if debug:
        # # plot displacement as scatter to make it easy to determine if enough trajectory points visually
        # plot_displacement_scatter(apprx_bspline_sampled, 'approx bspline')
        # plot_displacement_scatter(shftd_apprx_bspline, 'shftd approx bspline')
        # plot_displacement_scatter(trnsfrmd_apprx_bspline, 'transfrmd approx bspline')
        # plot_displacement_scatter(trnsfrmd_waypts, 'transfrmd waypts')
        # plot_displacement_scatter(trnsfrmd_waypts_sgnl_fltd, 'transfrmd waypts sgnl fltd bspline')
        plot_displacement_scatter(trnsfrmd_waypts_fltd[:,:2], 'transfrmd waypts fltd bspline')
        plot_displacement_scatter(applied_bspline_s[:,:2], 'transfrmd waypts fltd bspline')
        plot_curve_o3d(trnsfrmd_waypts_fltd[:, :3],
                    trnsfrmd_waypts_fltd[:, 3:])
        plot_curve_o3d(applied_bspline_s[:, :3],
                    applied_bspline_s[:, 3:])

    return applied_bspline


def contour_center_n_dists_frm_apprx(contour, apprx_contour):
    '''test'''
    cont_coord = contour.reshape(-1,2)
    #determine center of graphical input bspline
    cont_center = np.array([np.mean(cont_coord[:,0]),np.mean(cont_coord[:,1])])

    if apprx_contour is None:
        apprx_coord = contour.reshape(-1,2)
    else:
        apprx_coord = apprx_contour.reshape(-1,2)
    #determine L2 norm distance for every point on approximated bspline and center of bspline from graphical input
    dist_arr = np.zeros((apprx_coord.shape[0],1))
    for i in range(apprx_coord.shape[0]):
        dist_arr[i] = np.linalg.norm(cont_center - apprx_coord[i])

    return cont_center, dist_arr

def rotate_vector7D(points,pattern_rotation,plot_flag = True):
    ''' rotates points by pattern rotation'''
    # points: a set of vectors (x,y,z,quaternion) in 2D + rotation of quat around z axis
    # pattern_rotation: angle in degrees
    rospy.loginfo('Rotate pattern by {} degrees.'.format(pattern_rotation))
    theta = np.radians(pattern_rotation)
    c, s = np.cos(theta), np.sin(theta)
    R_mat = np.array(((c, -s), (s, c)))

    if plot_flag:
        # plot_displacement_scatter(points[:, 0:2], 'pattern_bspline_s')
        plot_quaternions(points[:, :3], points[:, 3:])  # plots the data before rotation

    # rotating x,y by rotation matrix
    points_rot2D = np.dot(R_mat, points[:, 0:2].T).T
    # plot_displacement_scatter(points_rot2D[:, 0:2], 'pattern_bspline_s')# plots the data after rotation

    # rotation of orientation: quat to Euler angles -> rotating around z by theta
    orig_quat = points[:, 3:]
    r = R.from_quat(orig_quat)
    orient_eulers = r.as_euler('zyx', degrees=True)
    orient_eulers[:, 0] = orient_eulers[:, 0] + pattern_rotation
    r = R.from_euler('zyx', orient_eulers, degrees=True)
    orient_quat = r.as_quat()
    points_rot3D = points.copy()
    points_rot3D[:, :2] = points_rot2D
    if plot_flag:
        # plot_quivers(points_rot3D[:, :3], quat2vec(orient_quat))  # plots the data after rotation
        plot_quaternions(points_rot3D[:, :3], orient_quat)  # plots the data before rotation

    return points_rot3D

def compute_apprx_spline_pattern(pattern_bspline_s, approx_method, nmbPts):
    'approximate spline for the pattern to use for shifting and rotating'
    # pattern_bspline_s: sampled pattern bspline
    # approx_method: approximation method do use

    # TODO: try to think up more robust way how to align along the given path -
    # none of Tan/linregression/zero y value method does work perfectly
    # TODO: show it on zero line

    if approx_method & ApproxMethod.TAN:
        apprx_xy = np.array([pattern_bspline_s[0,:2], pattern_bspline_s[1,:2],
                             pattern_bspline_s[-2,:2], pattern_bspline_s[-1,:2]])
    # apprx_xy = np.array([pattern_bspline_s[0], (pattern_bspline_s[0]+pattern_bspline_s[-1])/2, pattern_bspline_s[-1]])
    elif approx_method & ApproxMethod.REG:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            pattern_bspline_s[:, 0], pattern_bspline_s[:, 1])
        apprx_xy = np.array([[pattern_bspline_s[0, 0], intercept + slope * pattern_bspline_s[0, 0]], [pattern_bspline_s[1, 0], intercept + slope * pattern_bspline_s[1, 0]], [
                            pattern_bspline_s[-2, 0], intercept + slope * pattern_bspline_s[-2, 0]], [pattern_bspline_s[-1, 0], intercept + slope * pattern_bspline_s[-1, 0]]])
    elif approx_method & ApproxMethod.ZERO:
        apprx_xy = np.array([[pattern_bspline_s[0, 0], 0], [pattern_bspline_s[1, 0], 0], [
                            pattern_bspline_s[-2, 0], 0], [pattern_bspline_s[-1, 0], 0]])
    apprx_bspline = BsplineGen()
    apprx_bspline.generate_bspline_pars(apprx_xy, per_flag=False)

    # TODO: check that we have space to connect patterns (use_gap), so reduce nmbPts for apprx_bspline as well as for pattern and add connection points
    # apprx_bspline.nmbPts = (traj_bspline.shape[0] / no_of_prds) - 2*use_gap
    apprx_bspline.nmbPts = nmbPts
    apprx_bspline_sampled = apprx_bspline.generate_bspline_sample()
    # distance from center of the pattern spline
    input_bspline_center, dists_frm_apprx = contour_center_n_dists_frm_apprx(pattern_bspline_s[:, :2],
                                                                             apprx_bspline_sampled)
    # obtain "central" bspline by translating approximate bspline in the direction of the center of graphical input bspline
    shftd_apprx_bspline = apprx_bspline_sampled + (
                input_bspline_center - apprx_bspline_sampled[np.argmin(dists_frm_apprx)])
    return shftd_apprx_bspline

def resample_spline(points, nmbPts):
    res_bspline = BsplineGen()
    if points.shape[1]==7:
        res_bspline.generate_bspline_pars7D(np.transpose(points), per_flag=False)
    else:
        res_bspline.generate_bspline_pars(points, per_flag=False)
    res_bspline.nmbPts = nmbPts
    sampled_spline = res_bspline.generate_bspline_sample()
    return sampled_spline

if __name__ == "__main__":
    apply_pattern_to_contour()
