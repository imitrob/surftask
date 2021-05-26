import collections
import numpy as np
from scipy import signal, sparse, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import cv2
from cv_bridge import CvBridge

import rosbag
import rospy
import sensor_msgs.point_cloud2
from traj_complete_ros.utils import BsplineGen
from utils import smooth_contours_w_butterworth, set_up_2dplot, cont_pt_by_pt, plot_displacement_scatter
from scipy import stats


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


def apply_pattern_to_contour(pattern_bsplineP, traj_bspline, no_of_prds, use_gap=0, pattern_rotation=0, pattern_trim_start=0, pattern_trim_trail=0, approx_method=ApproxMethod.TAN):
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
    pattern_bsplineP.nmbPts = 92
    pattern_bspline_s = pattern_bsplineP.generate_bspline_sample()
    # plot_displacement_scatter(trnsfrmd_waypts, 'transfrmd waypts')

    # pattern rotation
    if pattern_rotation != 0:
        rospy.loginfo('Rotate pattern by {} degrees.'.format(pattern_rotation))
        theta = np.radians(pattern_rotation)
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array(((c, -s), (s, c)))
        pattern_bspline_s = np.dot(R_mat, pattern_bspline_s.T).T

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

    # TODO: add input parameters defined by user: trimming, gap, rotation of the pattern

    # pattern_bspline = generate_bspline_sample(pattern_bsplineP.nmbPts,pattern_bsplineP.pars,pattern_bsplineP.xout)
    # traj_bsplineS = traj_bspline

    # TODO: why are moved last M values before the first n-M values? Where M = n/number of repentitions (no_of_prds)
    traj_bspline = np.vstack(
        (traj_bspline[int(traj_bspline.shape[0] / no_of_prds):-1, :],
         traj_bspline[0:int(traj_bspline.shape[0] / no_of_prds), :]))

    # TODO: try to think up more robust way how to align along the given path -
    # none of Tan/linregression/zero y value method does work perfectly
    # TODO: add rotation of the pattern as a parameter defined by the user - then we can use zero line

    # approximate bspline using endpoints of graphical input bspline
    if approx_method & ApproxMethod.TAN:
        apprx_xy = np.array([pattern_bspline_s[0], pattern_bspline_s[1],
                             pattern_bspline_s[-2], pattern_bspline_s[-1]])
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
    # apprx_bspline.nmbPts = traj_bspline.shape[0] / no_of_prds
    # TODO: check
    apprx_bspline.nmbPts = (traj_bspline.shape[0] / no_of_prds) - 2*use_gap
    apprx_bspline_sampled = apprx_bspline.generate_bspline_sample()
    # apprx_bspline = generate_bspline(traj_bspline.shape[0] / no_of_prds, apprx_xy, per_flag=False)

    input_bspline_center, dists_frm_apprx = contour_center_n_dists_frm_apprx(pattern_bspline_s, apprx_bspline_sampled)

    # obtain "central" bspline by translating approximate bspline in the direction of the center of graphical input bspline
    shftd_apprx_bspline = apprx_bspline_sampled + (input_bspline_center - apprx_bspline_sampled[np.argmin(dists_frm_apprx)])
    pattern_bspline_r = BsplineGen()
    pattern_bspline_r.generate_bspline_pars(pattern_bspline_s, per_flag = False)
    pattern_bspline_r.nmbPts = (traj_bspline.shape[0] / no_of_prds)
    pattern_bspline_s = pattern_bspline_r.generate_bspline_sample()
    # TODO: check if the changes below are working to make desired repetitions and gaps

    # for manual implementation of constrained transformation using pseudoinverse
    M = np.zeros((shftd_apprx_bspline.shape[0] * 2, 4))
    for n in range(0, M.shape[0], 2):
        M[n, :] = np.hstack((shftd_apprx_bspline[n / 2, :], np.ones(1), np.zeros(1)))
        M[n + 1, :] = np.hstack(
            (shftd_apprx_bspline[n / 2, -1], -1.0 * shftd_apprx_bspline[n / 2, 0], np.zeros(1), np.ones(1)))

    trnsfrmd_apprx_bspline = np.zeros(traj_bspline.shape)
    trnsfrmd_waypts = np.zeros(
        (traj_bspline.shape[0], pattern_bspline_s.shape[1]))
    # TODO: check
    j_range = range(0, trnsfrmd_apprx_bspline.shape[0], shftd_apprx_bspline.shape[0] + 2*use_gap)
    k_range = range(0, trnsfrmd_waypts.shape[0], traj_bspline.shape[0]/no_of_prds)

    # beginning of the pattern
    j_range = range(use_gap, trnsfrmd_apprx_bspline.shape[0], shftd_apprx_bspline.shape[0] + 2*use_gap)
    # end of the pattern
    k_range = range(0, trnsfrmd_waypts.shape[0], traj_bspline.shape[0] / no_of_prds)


    print(j_range)
    print(k_range)
    for j, k in zip(j_range, k_range):

        A_r1 = np.hstack((np.cos(0), -np.sin(0), np.zeros(1)))
        A_r2 = np.hstack((np.sin(0), np.cos(0), np.zeros(1)))
        A_r1r2 = np.vstack((A_r1, A_r2))
        A_r3 = np.hstack((np.zeros(2), np.ones(1)))
        print(pattern_bspline_s.shape)
        print(traj_bspline.shape)
        print(shftd_apprx_bspline.shape)
        if int(cv2.__version__[0]) < 4:
            A_r1r2 = cv2.estimateRigidTransform(shftd_apprx_bspline.reshape(1, -1, 2),
                                            traj_bspline[j:j + shftd_apprx_bspline.shape[0], :].reshape(1, -1, 2),
                                            fullAffine=False)
        else:
            A_r1r2, _ = cv2.estimateAffinePartial2D(shftd_apprx_bspline.reshape(1, -1, 2),
                                            traj_bspline[j:j + shftd_apprx_bspline.shape[0], :].reshape(1, -1, 2))
        if A_r1r2 is None:
            print('Failed to estimate rigid transformation, trying manual implementation')
            print(j)
            print(k)
            A_elems = np.dot(np.linalg.pinv(M), traj_bspline[j:j + shftd_apprx_bspline.shape[0], :].reshape(-1, 1))

            # shear, coordinate flipping and non-uniform scaling constrained
            A_r1 = A_elems[0:3].reshape(-1)
            A_r2 = np.hstack((-1.0 * A_elems[1], A_elems[0], A_elems[3]))
            A_r1r2 = np.vstack((A_r1, A_r2))

        A_mat = np.vstack((A_r1r2, A_r3))
        trnsfrmd_apprx_bspline[j:j + shftd_apprx_bspline.shape[0], :] = cv2.transform(shftd_apprx_bspline.reshape(1, -1, 2), A_mat[0:2])[0]
        trnsfrmd_waypts[k:k + pattern_bspline_s.shape[0], :] = cv2.transform(pattern_bspline_s.reshape(1, -1, 2), A_mat[0:2])[0]
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
            print('')
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

    # smooth countour for shape matching,
    # WARNING: too low cutoff frequency causes start and end of contour to disconnect
  #  trnsfrmd_waypts_sgnl_fltd = smooth_contours_w_butterworth(trnsfrmd_apprx_bspline, 5, 0.4, 'low')
    trnsfrmd_waypts_sgnl_fltd = smooth_contours_w_butterworth(trnsfrmd_waypts, 5, 0.4, 'low')
    trnsfrmd_waypts_fltd = np.array(zip(list(trnsfrmd_waypts_sgnl_fltd.real), list(trnsfrmd_waypts_sgnl_fltd.imag)))

    applied_bspline = BsplineGen()
    applied_bspline.generate_bspline_pars(trnsfrmd_waypts_fltd, per_flag=True)
    applied_bspline.nmbPts = 2 * traj_bspline.shape[0]
    applied_bspline_s = applied_bspline.generate_bspline_sample()
    # applied_bspline = generate_bspline(2 * traj_bspline.shape[0], trnsfrmd_waypts_fltd, per_flag=True)

    # # plot displacement as scatter to make it easy to determine if enough trajectory points visually
    # plot_displacement_scatter(apprx_bspline_sampled, 'approx bspline')
    # plot_displacement_scatter(shftd_apprx_bspline, 'shftd approx bspline')
    # plot_displacement_scatter(trnsfrmd_apprx_bspline, 'transfrmd approx bspline')
    # plot_displacement_scatter(trnsfrmd_waypts, 'transfrmd waypts')
    # plot_displacement_scatter(trnsfrmd_waypts_sgnl_fltd, 'transfrmd waypts sgnl fltd bspline')
    # plot_displacement_scatter(trnsfrmd_waypts_fltd, 'transfrmd waypts fltd bspline')

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


if __name__ == "__main__":
    apply_pattern_to_contour()
