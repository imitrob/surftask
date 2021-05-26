#!/usr/bin/env python
import numpy as np
import yaml
import rospy
import os
from subprocess import Popen, PIPE, call
import pandas as pd
from datetime import datetime

path_to_experiment_table = os.path.expanduser("~/traj_complete_log/real_exp_half_circle/circle_knots_speed_series/experiments_real_exec_in_sim_v_vs_error.xls")
MAX_ATTEMPTS = 3

# load the table
experiments = pd.read_excel(path_to_experiment_table)
fields = experiments.columns

for idx, experiment in experiments.iterrows():
    dict_data = {k: v for k, v in zip(fields, experiment)}
    status = experiment["status"]
    if status == "done":
        print("Experiment {} already done, skipping.".format(experiment["name"]))
        continue
    if status == "hold":
        print("Experiment {} on hold, skipping.".format(experiment["name"]))
        continue

    print("Attempting experiment {}...".format(experiment["name"]))
    dt = datetime.now()
    stamp = dt.strftime("%Y-%m-%d_%H_%M_%S")
    dict_data["last_time"] = stamp

    experiment["robot_description"] = rospy.get_param('/robot_xml_path', default='robot.xml') # read robot xml file name from parameter server

    ret_val = -1
    attempts = 0
    row_data = yaml.safe_dump(dict_data)
    while ret_val != 0 and attempts < MAX_ATTEMPTS:
        # ret_val = call(["which", "roscore"])
        ret_val = call(["rosrun", "traj_complete_ros", "interface_node.py", "-e"] + [row_data])
        attempts += 1
    if attempts == MAX_ATTEMPTS:
        experiment["status"] = "failed"
        print("...{} failed!".format(experiment["name"]))
    else:
        experiment["status"] = "done"
        print("...{} succeeded!".format(experiment["name"]))
        experiment["last_time"] = stamp

    experiments.iloc[idx] = experiment
    experiments.to_excel(path_to_experiment_table, index=False)

experiments.to_excel(path_to_experiment_table, index=False)
