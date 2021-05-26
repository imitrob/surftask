import numpy as np
import json
import os
from matplotlib import pyplot as plt
import pandas as pd


def compute_angles_from_log(log_path):
    with open(log_path, "r") as f:
        d = json.load(f)

    curve_data = d["goal"]["curve"]
    curve = np.array([[c["x"], c["y"], c["z"]] for c in curve_data])
    return compute_angles(curve)


def compute_angles(xyz):
    A = xyz[:-2, :]
    B = xyz[1:-1, :]
    C = xyz[2:, :]
    U = B - A
    V = C - B
    U_norm = np.divide(U, np.linalg.norm(U, axis=1)[:, np.newaxis])
    V_norm = np.divide(V, np.linalg.norm(V, axis=1)[:, np.newaxis])
    U_norm[np.isnan(U_norm)] = 0
    V_norm[np.isnan(V_norm)] = 0
    dots = np.einsum("ij,ij->i", U_norm, V_norm)
    dots[np.isnan(dots)] = -1
    return np.arccos(dots)


def compute_histogram(angles):
    bins = [0, 10, 30, 45, 90, 180]
    mass, bins = np.histogram(np.rad2deg(angles), bins=bins)
    mass = mass / mass.astype(np.float).sum()
    return mass, bins


def plot_angles(mass, bins):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(mass)), mass, align="center", tick_label=[str(b) for b in bins[1:]])

    for i, v in enumerate(mass):
        ax.text(i, v + 0.002, "{:.2f}".format(v), color="black", ha='center')
    ax.set_xlabel("threshold angle (deg)")
    ax.autoscale(axis="y")
    ax.set_ylabel("fraction of total trajectory length")
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # load the table
    # / home / behrejan / traj_complete_log / experiments_v_vs_error.xls
    path_to_experiment_table = os.path.expanduser("~/traj_complete_log/experiments_v_vs_error.xls")

    # path_to_experiment_table = os.path.expanduser("~/traj_complete_log/experiments.xls")
    experiments = pd.read_excel(path_to_experiment_table)

    histograms = []
    bins = []
    # path = os.path.expanduser("~/traj_complete_log/Test_0_2020-10-31_21_25_04.json")
    for idx, experiment in experiments.iterrows():

        if not ("done" in str(experiment["status"]).lower()):
            continue
        try:
            if "skip" in str(experiment["skip_eval"]).lower():
                continue
        except ValueError:
            pass
        path = os.path.expanduser(os.path.join("~", "traj_complete_log", experiment["name"] + "_" + experiment["last_time"] + ".json"))
        angles = compute_angles_from_log(path)
        mass, bins = compute_histogram(angles)
        histograms.append(mass)
        bins = bins
        fig, ax = plot_angles(mass,           bins)
        fig.savefig(os.path.expanduser(os.path.join("~", "traj_complete_log", "chist_" + experiment["name"] + "_" + experiment["last_time"] + ".png")))
        # plt.show()

    if histograms:
        average_mass = np.vstack(histograms).mean(axis=0)
        fig, ax = plot_angles(average_mass, bins)
        fig.savefig(os.path.expanduser(os.path.join("~", "traj_complete_log", "chist_average_" + str(experiment["last_time"]) + ".png")))
