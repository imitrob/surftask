import numpy as np

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import pandas as pd

from traj_complete_ros.toppra_eef_vel_ct import plot_plan
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_time_index(time, times):
    return min(range(len(times)), key=lambda i: abs(times[i] - time))

def calc_dtw(ref_curve, exec_data):
    # dtw, d = similaritymeasures.dtw(ref_curve, exec_data)
    # df = similaritymeasures.frechet_dist(ref_curve, exec_data)
    # area = similaritymeasures.area_between_two_curves(ref_curve, exec_data)

    # x = np.array([1, 2, 3, 3, 7])
    # y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

    distance, path = fastdtw(ref_curve, exec_data, dist=euclidean)

    d2 = distance / len(path)
    #

    return distance, d2  # , area


def ref_vs_eef_graph(log_ana, ref_curve, result_folder, experiment={}, save_ref_curve=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print(log_ana.eef_pos)

    times = log_ana.joint_motion['t']

    def get_time_index(time, times):
        return min(range(len(times)), key=lambda i: abs(times[i] - time))

    t_start = log_ana.log['t'][log_ana.log['data'].index("start")]
    t_end = log_ana.log['t'][log_ana.log['data'].index("end")]
    idx_start = get_time_index(t_start + 1.0, times)
    idx_end = get_time_index(t_end - 1.0, times)

    ax.plot3D(log_ana.eef_pos[idx_start:idx_end, 0], log_ana.eef_pos[idx_start:idx_end, 1],
              log_ana.eef_pos[idx_start:idx_end, 2], 'red', label='End_effector Path')
    ax.plot3D(ref_curve[:, 0], ref_curve[:, 1], ref_curve[:, 2], 'gray', label='Reference Path')
    if not save_ref_curve is None:
        ax.plot3D(save_ref_curve[:, 0], save_ref_curve[:, 1], save_ref_curve[:, 2], 'blue', label='Safe Reference Path')

    ax.set_zlim([0.0, 0.03])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    # ax.axis('equal')
    vel_val = str(experiment['cart_vel_limit'])
    vel_val = vel_val.replace('.', '_')
    fig.legend()
    fig.tight_layout()
    fig.savefig(
        result_folder + '/eef_path_vs_ref_{}_{}_{}.pdf'.format(experiment['name'], experiment['default_pattern'],
                                                               vel_val))

def avg_acc(acc, w=None, times=None):
    if w is None:
        w = np.ones(7)
        w /= np.linalg.norm(w)
    if times is None:
        times = np.linspace(0.0, 1.0, acc.shape[0])

    delta_t = np.diff(times)
    delta_t = np.hstack([delta_t.mean(), delta_t])
    T = delta_t.sum()
    acc_abs = np.abs(acc)

    return (delta_t * np.linalg.norm(w * acc_abs, ord=1, axis=1)).sum() * (1 / T)


def weightedL2(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())

def MSRE(yi, ti, y0, a, b):
    ti = np.asarray(ti)
    idx_a = np.argmin(np.abs(ti-a)) +1
    idx_b = np.argmin(np.abs(ti-b)) -1
    if np.shape(yi) != np.shape(y0):
        y0 = np.ones(np.shape(yi)[0]) * y0
    e = (np.square(np.linalg.norm(yi[idx_a:idx_b,:3], axis=1) - y0[idx_a:idx_b])) # .mean(axis=None)
    e_mean = e.mean(axis=None)
    e_std = np.std(e)
    return e_mean, e_std