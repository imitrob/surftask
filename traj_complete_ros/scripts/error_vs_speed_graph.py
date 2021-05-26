
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from scipy.interpolate import interp1d


def get_value(table, row_column, row, x_column, y_column):
    loc = table.loc[table[row_column] == row]
    return float(loc[x_column]), float(loc[y_column])


def generate_names(base, s, e):
    return [base + "{:02d}".format(i) for i in range(s, e+1)]

# path_to_experiment_table_curve_eval = os.path.expanduser("~/traj_complete_log/experiments_v_vs_error.xls")
# courve_exps = pd.read_excel(path_to_experiment_table_curve_eval)

path_to_experiment_table_curve_eval = os.path.expanduser("~/traj_complete_log/experiments_v_vs_error_match_real.xls")
courve_exps = pd.read_excel(path_to_experiment_table_curve_eval)

path_to_experiment_table_real_and_sim = os.path.expanduser(
    "~/traj_complete_log/real_exp_half_circle/same_goal_vary_speed/experiments_real_exec_in_sim_v_vs_error.xls")

real_sim_exps = pd.read_excel(path_to_experiment_table_real_and_sim)

path_to_experiment_table_real_and_sim = os.path.expanduser(
    "~/traj_complete_log/real_exp_half_circle/same_goal_vary_speed/experiments_real_exec_in_sim_v_vs_error.xls")

path_to_experiment_table = os.path.expanduser(
    "~/traj_complete_log/real_exp_half_circle/circle_knots_speed_series/experiments_real_exec_in_sim_v_vs_error.xls")
real_sim_circle_exps = pd.read_excel(path_to_experiment_table)

loc = courve_exps.loc[courve_exps['name'] == "C0"]

print(loc)

names = {}

names['circle'] = generate_names('C', 0, 14)
names['circle_knot_artificial'] = generate_names('C_knot_', 0, 14)
names['circle_knot_sfr'] = generate_names('C_knot_sfr_', 0, 14)
names['circle_knot_f'] = generate_names('C_knot_f_', 0, 14)
# names['L_knot_desc'] = generate_names('E_desc_', 0, 14)
# names['L_knot_cart'] = generate_names('E_cart_', 0, 14)
names['circle_real'] = generate_names('circle_desc_', 1, 14)
names['circle_knot_sfr_real'] = generate_names('circle_knots_desc_', 1, 14)
names['C_knot_sfr_gap'] = generate_names('C_knot_sfr_gap_', 1, 14)

curves = {}

for name in names.keys():
    if name in ['L_knot_desc', 'L_knot_cart']:
        table = real_sim_exps
    elif name in ['circle_real', 'circle_knot_sfr_real']:
        table = real_sim_circle_exps
    else:
        table = courve_exps

    for n in names[name]:
        print(n)
        if name not in curves.keys():
            curves[name] = ([],[])
        x,y = get_value(table, 'name', n, 'cart_vel_limit', 'msre_vel')
        curves[name][0].append(x)
        curves[name][1].append(y)

w, h = plt.figaspect(0.5)
fig = plt.Figure(figsize=(w,h))
ax = fig.add_subplot(1,1,1)
# fig, ax = plt.subplots(1,1)
artists = {}
for name in names.keys():
    #
    ls = 'solid'
    c = 'r'
    if 'real' in name:
        ls = 'dashed'
    if 'sfr' in name:
        c = 'b'
    if name == 'circle' or name == 'circle_real':
        c = 'c'
    if name == 'circle_knot_f':
        c='y'
    if 'gap' in name:
        c='g'


    line = ax.plot(curves[name][0], curves[name][1], label=name, linestyle=ls, color=c, marker='x')
    artists[name] = line

ax.set_xlabel('End-Effector Speed [$m/s$]', fontsize=16)
ax.set_ylabel('$M_v$ (MSE) [$m^2/s^2$]', fontsize=16)
ax.hlines([0.005, 0.02], 0.0, 0.25, linestyles='dotted')

# ax.set_yscale('log')
# ax.set_aspect('auto')

fig.tight_layout()

def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

# show_figure(fig)
# fig.show()

custom_lines = [plt.Line2D([0], [0], color='b', lw=4, linestyle='dashed'),
                plt.Line2D([0], [0], color='y', lw=4, linestyle='solid'),
                plt.Line2D([0], [0], color='r', lw=4, linestyle='solid'),
                plt.Line2D([0], [0], color='b', lw=4, linestyle='solid'),
                plt.Line2D([0], [0], color='c', lw=4, linestyle='dashed'),
                plt.Line2D([0], [0], color='c', lw=4, linestyle='solid'),
                plt.Line2D([0], [0], color='g', lw=4, linestyle='solid')
                ]


fig.legend(custom_lines, ('circle_knot_sfr_real', 'circle_knot_f', 'circle_knot_artificial', 'circle_knot_sfr','circle_real', 'circle','C_knot_sfr_gap'), loc='upper center', fontsize=16)
# fig.legend(loc='upper center', fontsize=16)

# ax.set_aspect(0.2)

fig.savefig(os.path.expanduser("~/traj_complete_log/real_exp_half_circle/same_goal_vary_speed/error_speed_graph.pdf"))

# fig.show()

print('hallo')
