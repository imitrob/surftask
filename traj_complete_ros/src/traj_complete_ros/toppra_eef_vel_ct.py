from copy import copy, deepcopy

import rospy
import toppra as ta
from toppra.constraint import DiscretizationType
from trajectory_msgs.msg import JointTrajectoryPoint
from urdf_parser_py.urdf import URDF

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from traj_complete_ros.iiwa_fk import iiwa_jacobian
import numpy as np

from moveit_commander import MoveGroupCommander, RobotTrajectory


def get_joint_limits(joints):
    robot_urdf = URDF.from_parameter_server('robot_description')
    lower = []
    upper = []
    vel = []
    effort = []
    for joint in joints:
        lower += [robot_urdf.joint_map[joint].limit.lower]
        upper += [robot_urdf.joint_map[joint].limit.upper]
        vel += [robot_urdf.joint_map[joint].limit.velocity]
        effort += [robot_urdf.joint_map[joint].limit.effort]
    return lower, upper, vel, effort

def retime(plan, cart_vel_limit=-1.0, secondorder=True, pt_per_s=100, curve_len=None, start_delay=0.0):
    ta.setup_logging("INFO")
    assert isinstance(plan, RobotTrajectory)

    if not curve_len is None and cart_vel_limit > 0:
        n_grid = np.ceil(pt_per_s * curve_len / cart_vel_limit)
    else:
        n_grid = np.inf

    active_joints = plan.joint_trajectory.joint_names
    lower, upper, vel, effort = get_joint_limits(active_joints)

    # prepare numpy arrays with limits for acceleration
    alims = np.zeros((len(active_joints), 2))
    alims[:, 1] = np.array(len(lower) * [3.0])
    alims[:, 0] = np.array(len(lower) * [-3.0])

    # ... and velocity
    vlims = np.zeros((len(active_joints), 2))
    vlims[:, 1] = np.array(vel)
    vlims[:, 0] = (-1.0) * np.array(vel)

    use_cart_vel_limit = False
    if cart_vel_limit > 0:
        use_cart_vel_limit = True

    ss = [pt.time_from_start.to_sec() for pt in plan.joint_trajectory.points]
    way_pts = [list(pt.positions) for pt in plan.joint_trajectory.points]

    path = ta.SplineInterpolator(ss, way_pts)

    pc_vel = ta.constraint.JointVelocityConstraint(vlim=vlims)

    def vlims_func(val):
        eps = 0.001
        limit = cart_vel_limit
        J = iiwa_jacobian(path(val))
        direction = (path(val + eps) - path(val - eps)) / (2* eps)
        dir_norm = direction / np.linalg.norm(direction)
        x = limit / np.linalg.norm(np.dot(J, dir_norm))
        x = x * dir_norm

        x = np.abs(x)
        print("{}: {}".format(val, np.max(x)))
        lim = np.zeros((7, 2))
        lim[:, 1] = np.max(x)
        # if val <= 2.5:
        #     lim = np.zeros((7,2))
        #     lim[:,1] = np.max(x)
        # else:
        #     lim = np.zeros((7, 2))
        #     lim[:, 1] = np.array(7 * [1.0])

        lim[:, 0] = -lim[:,1]
        return lim

    pc_vel2 = ta.constraint.JointVelocityConstraintVarying(vlim_func=vlims_func)
    # pc_vel2.discretization_type = DiscretizationType.Interpolation

    pc_acc = ta.constraint.JointAccelerationConstraint(alim=alims)

    # def inv_dyn(q, qd, qgg):
    #     # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
    #     # velocities
    #     J = iiwa_jacobian(q)
    #     cart_vel = np.dot(J, qd)
    #     return np.linalg.norm(cart_vel)
    #
    # def g(q):
    #     return ([-0.2, 0.2])
    #
    # def F(q):
    #     return np.eye(1)
    #
    if secondorder:
        def my_inv_dyn(q, qd, qgg):
            # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
            # velocities
            J = iiwa_jacobian(q)
            cart_vel_sq = np.dot(np.dot(qd.T, J.T), np.dot(J, qd))

            print(cart_vel_sq)
            return np.array(len(qd) * [10000 * cart_vel_sq])

        def my_g(q):
            return np.array(len(q) * [10000 * cart_vel_limit**2])

        def my_F(q):
            return 1 * np.eye(len(q))

        eef_vel = ta.constraint.SecondOrderConstraint(inv_dyn=my_inv_dyn, constraint_F=my_F, constraint_g=my_g, dof=7,
                                                      discretization_scheme=DiscretizationType.Interpolation)
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path, solver_wrapper='seidel', gridpoints=np.linspace(0.0, ss[-1], np.min([int(n_grid), np.ceil(pt_per_s*ss[-1])])))
        # instance = ta.algorithm.TOPPRA([eef_vel], path, solver_wrapper='seidel')
    elif False:
        def my_inv_dyn(q, qd, qgg):
            # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
            # velocities
            J = iiwa_jacobian(q)
            cart_vel = np.dot(J, qd)

            print(np.linalg.norm(cart_vel))
            return np.array(len(qd) * [100 * np.linalg.norm(cart_vel)])

        def my_g(q):
            return np.array(len(q) * [100 * cart_vel_limit])

        def my_F(q):
            return 1 * np.eye(len(q))

        eef_vel = ta.constraint.SecondOrderConstraint(inv_dyn=my_inv_dyn, constraint_F=my_F, constraint_g=my_g, dof=7,
                                                      discretization_scheme=DiscretizationType.Collocation)
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path, solver_wrapper='seidel')
        # instance = ta.algorithm.TOPPRA([eef_vel], path, solver_wrapper='seidel')

    else:
        instance = ta.algorithm.TOPPRA([pc_vel, pc_vel2, pc_acc], path, gridpoints=np.linspace(0.0, ss[-1], np.min(n_grid, np.ceil(pt_per_s*ss[-1]))))

        # instance = ta.algorithm.TOPPRA([pc_vel, pc_vel2, pc_acc], path, gridpoints=np.linspace(0.0, ss[-1], 10000))

    # print(instance)
    # instance2 = ta.algorithm.TOPPRAsd([pc_vel, pc_acc], path)
    # instance2.set_desired_duration(60)
    jnt_traj = instance.compute_trajectory()
    feas_set = instance.compute_feasible_sets()

    # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
    ts_sample = np.linspace(0, jnt_traj.duration, np.ceil(100 * jnt_traj.duration))
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)

    new_plan = deepcopy(plan)
    new_plan.joint_trajectory.points = []

    for t, q, qd, qdd in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
        pt = JointTrajectoryPoint()
        pt.time_from_start = rospy.Duration.from_sec(t + start_delay)
        pt.positions = q
        pt.velocities = qd
        pt.accelerations = qdd
        new_plan.joint_trajectory.points.append(pt)

    if rospy.get_param('plot_joint_trajectory', default=False):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, sharex=True)
        for i in range(path.dof):
            # plot the i-th joint trajectory
            axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
            axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
            axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        plt.show()

    return new_plan

def retime_old(plan, cart_vel_limit=-1.0):
    ta.setup_logging("INFO")
    assert isinstance(plan, RobotTrajectory)

    active_joints = plan.joint_trajectory.joint_names
    lower, upper, vel, effort = get_joint_limits(active_joints)

    # prepare numpy arrays with limits for acceleration
    alims = np.zeros((len(active_joints), 2))
    alims[:,1] = np.array(len(lower) * [4.0])
    alims[:, 0] = np.array(len(lower) * [-4.0])

    # ... and velocity
    vlims = np.zeros((len(active_joints), 2))
    vlims[:,1] = np.array(vel)
    vlims[:, 0] = (-1.0) * np.array(vel)

    use_cart_vel_limit = False
    if cart_vel_limit > 0:
        use_cart_vel_limit = True


    ss = [pt.time_from_start.to_sec() for pt in plan.joint_trajectory.points]
    way_pts = [list(pt.positions) for pt in plan.joint_trajectory.points]

    path = ta.SplineInterpolator(ss, way_pts)

    pc_vel = ta.constraint.JointVelocityConstraint(vlim=vlims)

    pc_acc = ta.constraint.JointAccelerationConstraint(alim=alims)

    # def inv_dyn(q, qd, qgg):
    #     # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
    #     # velocities
    #     J = iiwa_jacobian(q)
    #     cart_vel = np.dot(J, qd)
    #     return np.linalg.norm(cart_vel)
    #
    # def g(q):
    #     return ([-0.2, 0.2])
    #
    # def F(q):
    #     return np.eye(1)
    #
    if use_cart_vel_limit:
        def my_inv_dyn(q, qd, qgg):
            # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
            # velocities
            J = iiwa_jacobian(q)
            cart_vel = np.dot(J, qd)

            print(np.linalg.norm(cart_vel))
            return np.array(len(qd) * [100* np.linalg.norm(cart_vel)])

        def my_g(q):
            return np.array(len(q) * [100*cart_vel_limit])

        def my_F(q):
            return 1 * np.eye(len(q))

        eef_vel = ta.constraint.SecondOrderConstraint(inv_dyn=my_inv_dyn, constraint_F=my_F, constraint_g=my_g, dof=7, discretization_scheme=DiscretizationType.Collocation)
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path, solver_wrapper='seidel')
        # instance = ta.algorithm.TOPPRA([eef_vel], path, solver_wrapper='seidel')

    else:
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc], path)

    # print(instance)
    # instance2 = ta.algorithm.TOPPRAsd([pc_vel, pc_acc], path)
    # instance2.set_desired_duration(60)
    jnt_traj = instance.compute_trajectory()
    feas_set = instance.compute_feasible_sets()


    # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
    ts_sample = np.linspace(0, jnt_traj.duration, np.ceil(100 * jnt_traj.duration))
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)

    new_plan = deepcopy(plan)
    new_plan.joint_trajectory.points = []

    for t, q, qd, qdd in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
        pt = JointTrajectoryPoint()
        pt.time_from_start = rospy.Duration.from_sec(t)
        pt.positions = q
        pt.velocities = qd
        pt.accelerations = qdd
        new_plan.joint_trajectory.points.append(pt)

    if rospy.get_param('plot_joint_trajectory', default=False):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, sharex=True)
        for i in range(path.dof):
            # plot the i-th joint trajectory
            axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
            axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
            axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        plt.show()

    return new_plan

def plan_2_arr(plan):
    ss = np.array([pt.time_from_start.to_sec() for pt in plan.joint_trajectory.points])
    way_pts = np.array([list(pt.positions) for pt in plan.joint_trajectory.points])
    vel = np.array([list(pt.velocities) for pt in plan.joint_trajectory.points])
    acc = np.array([list(pt.accelerations) for pt in plan.joint_trajectory.points])

    eef_vel = np.array([np.linalg.norm(np.dot(iiwa_jacobian(pt),v)) for pt, v in zip(way_pts, vel)])
    return ss, way_pts, vel, acc, eef_vel

def plot_motion(ss, way_pts, vel, acc=None, title='', save='', show=True, ret=False, v0=None):
    eef_vel = np.array([np.linalg.norm(np.dot(iiwa_jacobian(pt), v)) for pt, v in zip(way_pts, vel)])
    have_acc = True
    if acc is None:
        have_acc = False

    if have_acc:
        fig, axs = plt.subplots(4, 1, sharex=True)
        pos_ax = axs[0]
        vel_ax = axs[1]
        acc_ax = axs[2]
        eef_vel_ax = axs[3]
    else:
        w, h = plt.figaspect(0.5)
        fig = plt.Figure(figsize=(w, h))
        ax = fig.add_subplot(3, 1, 1)
        pos_ax = fig.add_subplot(3, 1, 1)
        vel_ax = fig.add_subplot(3, 1, 2)
        eef_vel_ax = fig.add_subplot(3, 1, 3)

        # fig, axs = plt.subplots(3, 1, sharex=True)
        # pos_ax = axs[0]
        # vel_ax = axs[1]
        # eef_vel_ax = axs[2]
    fig.suptitle(title, fontsize=16)
    for i in range(way_pts.shape[1]):
        # plot the i-th joint trajectory
        pos_ax.plot(ss, way_pts[:, i], c="C{:d}".format(i), label='Joint {}'.format(i))
        vel_ax.plot(ss, vel[:, i], c="C{:d}".format(i))
        if have_acc:
            acc_ax.plot(ss, acc[:, i], c="C{:d}".format(i))
    eef_vel_ax.plot(ss, eef_vel, c="r", label='eef speed')
    pos_ax.set_ylabel("Position (rad)")
    vel_ax.set_ylabel("Velocity (rad/s)")
    if have_acc:
        acc_ax.set_ylabel("Acceleration (rad/s2)")
    eef_vel_ax.set_ylabel("EEF Speed (m/s)")
    eef_vel_ax.set_xlabel("Time (s)")
    if v0 is not None:
        eef_vel_ax.hlines(v0, ss[0]-1.0, ss[-1]+1.0, linestyles='dotted', label='v0')
    fig.legend()
    fig.tight_layout()
    if len(save) > 0:
        import os
        dir = os.path.split(save)[0]
        os.path.isdir(dir)

        fig.savefig(fname=save)
    if show:
        plt.show()
    if ret:
        return fig, axs

def plot_plan(plan, title='', save='', show=True, ret=False):

    ss = np.array([pt.time_from_start.to_sec() for pt in plan.joint_trajectory.points])
    way_pts = np.array([list(pt.positions) for pt in plan.joint_trajectory.points])
    vel = np.array([list(pt.velocities) for pt in plan.joint_trajectory.points])
    acc = np.array([list(pt.accelerations) for pt in plan.joint_trajectory.points])

    eef_vel = np.array([np.linalg.norm(np.dot(iiwa_jacobian(pt),v)) for pt, v in zip(way_pts, vel)])


    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.suptitle(title, fontsize=16)
    for i in range(way_pts.shape[1]):
        # plot the i-th joint trajectory
        axs[0].plot(ss, way_pts[:, i], c="C{:d}".format(i))
        axs[1].plot(ss, vel[:, i], c="C{:d}".format(i))
        axs[2].plot(ss, acc[:, i], c="C{:d}".format(i))
    axs[3].plot(ss, eef_vel, c="r")
    axs[2].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (rad)")
    axs[1].set_ylabel("Velocity (rad/s)")
    axs[2].set_ylabel("Acceleration (rad/s2)")
    axs[3].set_ylabel("EEF Speed (m/s)")
    if len(save) > 0:
        import os
        dir = os.path.split(save)[0]
        os.path.isdir(dir)
        fig.savefig(fname=save)
    if show:
        plt.show()
    if ret:
        return fig, axs





if __name__ == "__main__":
    # run in sourced terminal:
    # roslaunch capek_launch virtual_robot.launch
    rospy.init_node('toppra_testing')

    mg = MoveGroupCommander('r1_arm')

    # plan to move the eef 20 cm down
    current_pose = mg.get_current_pose(mg.get_end_effector_link())
    target_pose = deepcopy(current_pose)
    target_pose.pose.position.z -= 0.2
    plan = mg.plan(target_pose)

    # plot plan as it is produced by MoveIt!
    plot_plan(plan)

    # retime with Kuka limits on joint velocities and accelerations
    plan2 = retime(plan)
    plot_plan(plan2)

    # retime with additional constraint on the cartesian eef velocity
    plan3 = retime(plan, cart_vel_limit=0.1)
    plot_plan(plan3)

    print('done')

