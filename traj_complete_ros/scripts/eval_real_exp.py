import datetime

import rospy

from traj_complete_ros.log_analyzer import LogAnalyzer
from traj_complete_ros.measures import ref_vs_eef_graph, MSRE, calc_dtw, get_time_index
import numpy as np
from traj_complete_ros.just_send_goal_interface import UI
from traject_msgs.msg import CurveExecutionGoal
from rospy_message_converter import message_converter

from traj_complete_ros.toppra_eef_vel_ct import plot_motion

import pandas as pd

def make_safe_ref_curve(curve, Z_MIN):
    mask = np.where(curve[:, 2] < Z_MIN)
    curve[:, 2][mask] = Z_MIN
    return curve

def do_exps(goal, path_to_experiment_table='/home/behrejan/traj_complete_log/real_exp_half_circle/same_goal_vary_speed/experiments_real_exec_in_sim_v_vs_error.xls'):
    experiments = pd.read_excel(path_to_experiment_table)
    fields = experiments.columns
    assert isinstance(goal, CurveExecutionGoal)

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
        dt = datetime.datetime.now()
        stamp = dt.strftime("%Y-%m-%d_%H_%M_%S")
        experiment["last_time"] = stamp

        experiment["robot_description"] = rospy.get_param('/robot_xml_path',
                                                          default='robot.xml')  # read robot xml file name from parameter server

        log_name = str(experiment["name"]) + "_" + experiment["last_time"]


        # run the exp
        goal.opt.cart_vel_limit = experiment["cart_vel_limit"]
        ui = UI(goal=goal, log_name='{}/{}'.format(str(experiment["name"]), log_name))
        success = ui.get_state()

        if success < 1:
            experiment["status"] = "failed"
            print("...{} failed!".format(experiment["name"]))
        else:
            experiment["status"] = "done"
            print("...{} succeeded!".format(experiment["name"]))
            experiment["last_time"] = stamp

        experiments.iloc[idx] = experiment
        experiments.to_excel(path_to_experiment_table, index=False)

def get_exp_str(experiment, name, folder, ending=''):
    ending = ending.replace('.', '')
    vel_val = str(experiment['cart_vel_limit']).replace('.', '_')
    return "{}/{}_{}_{}_{}_exec{}.".format(folder, name, experiment['name'],experiment['default_pattern'],vel_val, int(experiment['executor_engine'])) + ending


def main():
    urdf = '/home/behrejan/traj_complete_log/real_exp_half_circle/robot_14b3883547cd46feb28f53f52a245683.xml'
    data = '/home/behrejan/traj_complete_log/real_exp_half_circle/2021_02_20_23_25_46.json'
    result_folder = '/home/behrejan/traj_complete_log/real_exp_half_circle'
    exp = {'cart_vel_limit': 0.05, 'default_pattern': 'knot', 'name': 'real_L_knot', 'executor_engine': 3}
    logana = LogAnalyzer(urdf_file=urdf, eef='tool0')
    logana.load_data(data)
    logana.check_heal_data()
    logana.check_heal_real_data()
    logana.calc_eef_poses()

    ref_curve = logana.get_ref_curve()
    save_ref_curve = np.copy(ref_curve)
    make_safe_ref_curve(save_ref_curve, 0.02)

    times = logana.joint_motion['t']
    t_start = logana.log['t'][logana.log['data'].index("start")]
    t_end = logana.log['t'][logana.log['data'].index("end")]
    idx_start = get_time_index(t_start + 1.0, times)
    idx_end = get_time_index(t_end - 1.0, times)


    ref_vs_eef_graph(logana, ref_curve, result_folder, exp, save_ref_curve=save_ref_curve)


    dtw, dtw_normed = calc_dtw(save_ref_curve, logana.eef_pos[idx_start:idx_end, :3])
    exp['dtw'] = dtw
    exp['dtw_normed'] = dtw_normed


    exp['msre_vel'] = \
    MSRE(logana.eef_vel, logana.joint_motion['t'], exp['cart_vel_limit'], logana.get_log_time(u'start'),
         logana.get_log_time(u'end'))[0]

    start = logana.get_log_time('start')
    end = logana.get_log_time('end')
    ss, way_pts, vel, acc = logana.get_joint_motion_as_arr(start, end)
    plot_motion(ss, way_pts, vel, None, save=result_folder + '/motion.pdf', v0=exp['cart_vel_limit'])

    traveled_distance = logana.traveled_distance(trimmed=True, extra_trim=[0, 0])
    for i, dist in enumerate(traveled_distance):
        exp["traveled_distance_{}".format(i)] = dist

    # TODO: should this be 2 norm or sum of all angles
    exp["traveled_distance_norm"] = np.linalg.norm(traveled_distance, ord=1)

    exp['avg_acc'] = np.linalg.norm(logana.get_acc_distr(extra_trim=[1, 1], save=get_exp_str(exp, 'acc_histogram', result_folder, 'pdf')))

    print(exp)

    # print(ref_curve)
    goal_dict = logana.get_goal()
    # goal = CurveExecutionGoal()
    # goal.
    goal = message_converter.convert_dictionary_to_ros_message('traject_msgs/CurveExecutionGoal', goal_dict)
    # goal.opt.cart_vel_limit = 0.05

    # do_exps(goal)

    # ui = UI(goal=goal)

    print('hallo freund')


if __name__ == "__main__":
    main()