import numba

import numpy as np
import json
import os

import similaritymeasures
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd

from distutils import dir_util

from traj_complete_ros.toppra_eef_vel_ct import plot_plan


# def compute_angles_from_log(log_path):
#     with open(log_path, "r") as f:
#         d = json.load(f)
#
#     curve_data = d["goal"]["curve"]
#     curve = np.array([[c["x"], c["y"], c["z"]] for c in curve_data])
#     return compute_angles(curve)
#
#
# def compute_angles(xyz):
#     A = xyz[:-2, :]
#     B = xyz[1:-1, :]
#     C = xyz[2:, :]
#     U = B - A
#     V = C - B
#     U_norm = np.divide(U, np.linalg.norm(U, axis=1)[:, np.newaxis])
#     V_norm = np.divide(V, np.linalg.norm(V, axis=1)[:, np.newaxis])
#     U_norm[np.isnan(U_norm)] = 0
#     V_norm[np.isnan(V_norm)] = 0
#     dots = np.einsum("ij,ij->i", U_norm, V_norm)
#     dots[np.isnan(dots)] = -1
#     return np.arccos(dots)
#
#
# def compute_histogram(angles):
#     bins = [0, 10, 30, 45, 90, 180]
#     mass, bins = np.histogram(np.rad2deg(angles), bins=bins)
#     mass = mass / mass.astype(np.float).sum()
#     return mass, bins
#
#
# def plot_angles(mass, bins):
#     fig, ax = plt.subplots()
#     ax.bar(np.arange(len(mass)), mass, align="center", tick_label=[str(b) for b in bins[1:]])
#
#     for i, v in enumerate(mass):
#         ax.text(i, v + 0.002, "{:.2f}".format(v), color="black", ha='center')
#     ax.set_xlabel("threshold angle (deg)")
#     ax.autoscale(axis="y")
#     ax.set_ylabel("fraction of total trajectory length")
#     fig.tight_layout()
#     return fig, ax
from traj_complete_ros.log_analyzer import LogAnalyzer
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
#
# x = np.array([1, 2, 3, 3, 7])
# y = np.array([1, 2, 2, 2, 2, 2, 2, 4])
#
# distance, path = fastdtw(x, y, dist=euclidean)
#
# print(distance)
# print(path)


# @numba.jit
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

def plot_speed(times, eef_pos, eef_vel):
    fig, ax = plt.subplots()


    print('hallo')
    speed = eef_vel

    speeds = [np.linalg.norm(v[:3]) for v in eef_vel]
    # ax.plot(times, np.linalg.norm(eef_vel, axis=0))
    ax.plot(times, speeds)


    ax.set_xlabel("threshold angle (deg)")
    ax.autoscale(axis="y")
    ax.set_ylabel("fraction of total trajectory length")
    fig.tight_layout()
    return fig, ax


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

def msre_integration(yi, ti, y0, a, b):
    from scipy.integrate import quad
    from scipy.interpolate import interpolate
    ti = np.asarray(ti)

    y_abs = np.linalg.norm(yi[:, :3], axis=1).reshape((-1,1))
    # fun = interpolate.interp1d(ti, np.linalg.norm(yi[:, :3], axis=1))
    fun = interpolate.interp1d(ti, y_abs, axis=0)
    # fun = interpolate.interp1d(np.array(ti), y_abs)
    # def fun(t):
    #     np.argmin()

    def integrand(t):
        # return np.square(np.linalg.norm(fun(t)[0:3]) - y0)
        return fun(t)

    return quad(integrand, a, b)

def get_exp_str(experiment, name, folder, ending=''):
    ending = ending.replace('.', '')
    vel_val = str(experiment['cart_vel_limit']).replace('.', '_')
    return "{}/{}_{}_{}_{}_exec{}.".format(folder, name, experiment['name'],experiment['default_pattern'],vel_val, int(experiment['executor_engine'])) + ending


if __name__ == "__main__":
    # load the table
    path_to_experiment_table = os.path.expanduser("~/traj_complete_log/experiments.xls")
    experiments = pd.read_excel(path_to_experiment_table)

    histograms = []
    bins = []
    # path = os.path.expanduser("~/traj_complete_log/Test_0_2020-10-31_21_25_04.json")
    for idx, experiment in experiments.iterrows():
        print('Evaluating experiment {}.'.format(experiment['name']))

        if not ("done" in str(experiment["status"]).lower()):
            print('Experiment not yet executed.')
            continue
        try:
            if "skip" in str(experiment["skip_eval"]).lower():
                print('Experiment evaluation skipped. Remove skip flag in experiment table to run eval.')
                continue
        except ValueError:
            pass

        # if not ("rect_9" in str(experiment["name"]).lower()):
        #     continue

        if "" == str(experiment["robot_description"]).lower():
            robotxml = "robot.xml"
        else:
            robotxml = str(experiment["robot_description"])

        path = os.path.expanduser(os.path.join("~", "traj_complete_log", str(experiment["name"]) + "_" + experiment["last_time"] + ".json"))

        result_folder = os.path.join(os.path.split(path)[0], str(experiment['name']))
        dir_util.mkpath(result_folder)




        urdf_path = os.path.expanduser(os.path.join("~", "traj_complete_log", robotxml))
        assert os.path.isfile(urdf_path)
        # log_ana = LogAnalyzer(urdf_path, world_frame='world', eef='panda_link8')
        log_ana = LogAnalyzer(urdf_path, world_frame='world', eef='tool_r2')
        log_ana.load_data(path)
        log_ana.check_heal_data()
        log_ana.calc_eef_poses()

        ref_curve = log_ana.get_ref_curve()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        print(log_ana.eef_pos)

        times = log_ana.joint_motion['t']


        def get_time_index(time, times):
            return min(range(len(times)), key=lambda i: abs(times[i] - time))


        t_start = log_ana.log['t'][log_ana.log['data'].index("start")]
        t_end = log_ana.log['t'][log_ana.log['data'].index("end")]
        idx_start = get_time_index(t_start+1.0, times)
        idx_end = get_time_index(t_end-1.0, times)

        ax.plot3D(log_ana.eef_pos[idx_start:idx_end,0], log_ana.eef_pos[idx_start:idx_end,1], log_ana.eef_pos[idx_start:idx_end,2], 'gray')
        ax.plot3D(ref_curve[:,0], ref_curve[:,1], ref_curve[:,2], 'red')

        vel_val = str(experiment['cart_vel_limit'])
        vel_val = vel_val.replace('.', '_')
        fig.savefig(result_folder + '/eef_path_vs_ref_{}_{}_{}.png'.format(experiment['name'],experiment['default_pattern'],vel_val))

        plot_plan(log_ana.get_log_as_plan(extra_trim=[1,1]), 'Joint Motion and End-Effector Speed', save=result_folder + "/Joint_motion_eef_vel_{}_{}_{}_{}.png".format(experiment['name'],experiment['default_pattern'],vel_val, experiment['executor_engine']), show=False)


        # fig.show()

        dists = calc_dtw(ref_curve, log_ana.eef_pos[idx_start:idx_end, :3])
        experiment['dtw'] = dists[0]
        experiment['dtw_normalized'] = dists[1]
        # experiment['frechet'] = dists[1]

        # plot_speed(times, log_ana.eef_pos, log_ana.eef_vel)

        # experiment['msre_vel']
        experiment['msre_vel'] = MSRE(log_ana.eef_vel, log_ana.joint_motion['t'], experiment['cart_vel_limit'], log_ana.get_log_time(u'start'), log_ana.get_log_time(u'end'))[0]
        # experiment['msre_vel']
        # msre_int = msre_integration(log_ana.eef_vel, log_ana.joint_motion['t'], experiment['cart_vel_limit'], log_ana.get_log_time(u'start'), log_ana.get_log_time(u'end'))

        traveled_distance = log_ana.traveled_distance(trimmed=True, extra_trim=[1,1])
        for i, dist in enumerate(traveled_distance):
            experiment["traveled_distance_{}".format(i)] = dist

        # TODO: should this be 2 norm or sum of all angles
        experiment["traveled_distance_norm"] = np.linalg.norm(traveled_distance, ord=1)

        experiment['avg_acc'] = np.linalg.norm(log_ana.get_acc_distr(extra_trim=[1,1], save=get_exp_str(experiment, 'acc_histogram', result_folder, 'pdf')))

        print('hallo')

        experiments.iloc[idx] = experiment

    experiments.to_excel(path_to_experiment_table, index=False)

        # angles = compute_angles_from_log(path)
        # mass, bins = compute_histogram(angles)
        # histograms.append(mass)
        # bins = bins
        # fig, ax = plot_angles(mass,           bins)
        # fig.savefig(os.path.expanduser(os.path.join("~", "traj_complete_log", "chist_" + experiment["name"] + "_" + experiment["last_time"] + ".png")))
        # plt.show()

    # if histograms:
    #     average_mass = np.vstack(histograms).mean(axis=0)
    #     fig, ax = plot_angles(average_mass, bins)
    #     fig.savefig(os.path.expanduser(os.path.join("~", "traj_complete_log", "chist_average_" + experiment["last_time"] + ".png")))
