

import json
import numpy as np
import rospy
from moveit_msgs.msg import RobotTrajectory
from tf.transformations import translation_matrix
from traject_msgs.msg import CurveExecutionGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from urdf_parser_py.urdf import URDF

from pykdl_utils.kdl_kinematics import KDLKinematics
from rospy_message_converter import message_converter

from scipy import signal, interpolate
from scipy.spatial.distance import euclidean
from traj_complete_ros.measures import get_time_index



class LogAnalyzer(object):
    # joint state -> compute eef pose and velocity, acc, jerk
    # joint pos, vel, acc, jerk
    # deviation from the curve
    # TODO: how to get this robot dependent information?
    # w2table2 = translation_matrix(np.array([0.42783, -0.85741, -0.0054846]))
    # w2table2 = translation_matrix(np.array([0.3, 0.0, 0.0]))

    # This is the correct transform
    # w2table2 = translation_matrix(np.array([0.4275, -0.85483, -0.005]))

    w2table2 = translation_matrix(np.array([0.4275, -0.85483, -0.0]))
    w2table1 = translation_matrix(np.array([0.4275, -0.0, -0.0]))




    def __init__(self, urdf_file, world_frame='world', eef='r2_link_tip'):

        # load kdl
        # eef = 'r2_link_tip'
        self._eef = eef
        self._world_frame = world_frame
        # self.robot_urdf = URDF.from_parameter_server('robot_description')
        self.robot_urdf = URDF.from_xml_file(urdf_file)

        self.kdl_kin = KDLKinematics(self.robot_urdf, self._world_frame, end_link=eef)

        self.joint_names = []

        self.joint_motion = {'t': [],
                             'pos': [],
                             'vel': [],
                             'acc': []}
        self.eef_pos = None
        self.eef_vel = None

        # two tables: xyz curve points
        # sec table: t: vel, acc, pos for eef and also for joints


        self.log = {'t': [],
                    'data': []}

    def get_joint_limits(self, joints=None):
        robot_urdf = self.robot_urdf
        if joints is None:
            joints = self.joint_names
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

    def get_acc_limit(self, joints=None):
        lower, upper, vel, effort = self.get_joint_limits(joints)
        # prepare numpy arrays with limits for acceleration
        alims = np.zeros((len(self.joint_names), 2))
        alims[:, 1] = np.array(len(lower) * [3.0])
        alims[:, 0] = np.array(len(lower) * [-3.0])
        return alims

    def get_joint_motion_as_arr(self, start_time, end_time):
        ss = np.array(self.joint_motion['t'])
        start_idx = get_time_index(start_time, ss)
        end_idx = get_time_index(end_time, ss)
        pos = np.array(self.joint_motion['pos'])
        vel = np.array(self.joint_motion['vel'])
        acc = np.array(self.joint_motion['acc'])
        return ss[start_idx:end_idx], pos[start_idx:end_idx], vel[start_idx:end_idx], acc[start_idx:end_idx]

    def get_log_time(self, key):
        return self.log['t'][self.log['data'].index(key)]

    def get_trim_time(self, trimmed=True, extra_trim=None, idx=False):
        if extra_trim is None:
            extra_trim = [0, 0]

        if trimmed:
            start = self.get_log_time('start') + extra_trim[0]
            end = self.get_log_time('end') - extra_trim[1]
        else:
            start = self.joint_motion['t'][0]
            end = self.joint_motion['t'][-1]

        if idx:
            start_idx = np.argmin(np.abs(np.array(self.joint_motion['t']) - start))
            end_idx = np.argmin(np.abs(np.array(self.joint_motion['t']) - end))
            return start_idx, end_idx
        else:
            return start, end

    def get_log_as_plan(self, trimmed=True, extra_trim=None):
        if extra_trim is None:
            extra_trim = [0, 0]
        start = self.get_log_time('start') + extra_trim[0]
        end = self.get_log_time('end') - extra_trim[1]
        plan = RobotTrajectory()
        plan.joint_trajectory.joint_names = self.joint_names
        for t, pos, vel, acc in zip(self.joint_motion['t'], self.joint_motion['pos'], self.joint_motion['vel'], self.joint_motion['acc']):
            if t < start and trimmed:
                continue
            elif t > end and trimmed:
                break
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration.from_sec(t)
            p.positions = pos
            p.velocities = vel
            p.accelerations = acc
            plan.joint_trajectory.points.append(p)

        return plan

    def get_acc_distr(self, trimmed=True, extra_trim=None, save=''):
        from matplotlib import pyplot as plt
        idx_start, idx_end = self.get_trim_time(trimmed=trimmed, extra_trim=extra_trim, idx=True)
        acc = np.array(self.joint_motion['acc'])[idx_start:idx_end]
        acc_abs = np.abs(acc)
        print("hallo")
        hist = np.histogram(acc, 30)
        fig = plt.figure()
        ax = plt.axes()
        ax.hist(acc, 51)
        if save.__len__() > 0:
            fig.savefig(save)
        # fig.show()
        return acc_abs.mean(axis=0)

    def get_acc(self, trimmed=True, extra_trim=None):
        idx_start, idx_end = self.get_trim_time(trimmed=trimmed, extra_trim=extra_trim, idx=True)
        acc = np.array(self.joint_motion['acc'])[idx_start:idx_end]
        # acc_abs = np.abs(acc)

        return acc

    def traveled_distance(self, trimmed=True, extra_trim=None, min_dist=0.001):
        if extra_trim is None:
            extra_trim = [0, 0]
        start = self.get_log_time('start') + extra_trim[0]
        end = self.get_log_time('end') - extra_trim[1]

        s = np.zeros(self.joint_names.__len__())
        last_point = None
        for t, pos, vel, acc in zip(self.joint_motion['t'], self.joint_motion['pos'], self.joint_motion['vel'], self.joint_motion['acc']):
            if t < start and trimmed:
                continue
            elif t > end and trimmed:
                break
            if last_point is None:
                last_point = np.array(pos)
                continue
            dist = euclidean(last_point, np.array(pos))
            # dist = np.abs(last_point - np.array(pos))
            if np.linalg.norm(dist) < min_dist:
                continue
            s += np.abs(last_point - np.array(pos))
            last_point = np.array(pos)

        return s

    def load_data(self, file):
        with open(file, 'r') as f:
            data = json.load(f)

        self.joint_motion = data['motion']
        self.log = data['log']
        self.joint_names = data['joint_names']
        # self.ref_traj = message_converter.convert_dictionary_to_ros_message(CurveExecutionGoal, data['goal'])
        # goal_dict = data['goal']
        # message_converter.convert_dictionary_to_ros_message(CurveExecutionGoal, goal_dict)
        self.ref_traj = data['goal']

    def check_heal_data(self):
        if max(self.joint_motion['vel']) == min(self.joint_motion['vel']):
            # velocity data is not there. let's fill it via numerical differentiation
            x = self.joint_motion['t']
            dx = np.diff(x, prepend=0.0).reshape((-1,1))
            dxdx = dx ** 2
            f = np.array(self.joint_motion['pos'])
            # First derivatives:
            df = np.diff(f, axis=0, prepend=f[0].reshape((1, -1))) / dx
            cf = np.convolve(f[0][1:-1], [1, -1]) / dx
            # gf = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap') / dx
            df_norm = np.linalg.norm(df, axis=1)

            # Second derivatives:
            ddf = np.diff(f, n=2, axis=0, prepend=f[0].reshape((1, -1)), append=f[-1].reshape((1, -1))) / dxdx
            ccf = np.convolve(f[0][1:-2], [1, -2, 1]) / dxdx
            # ggf = ndimage.gaussian_filter1d(f, sigma=1, order=2, mode='wrap') / dxdx

            self.joint_motion['vel'] = df
            self.joint_motion['acc'] = ddf
            return

    def check_heal_real_data(self):
        # if max(self.joint_motion['vel']) == min(self.joint_motion['vel']):
        # velocity data is not there. let's fill it via numerical differentiation
        x = self.joint_motion['t']
        dx = np.diff(x, prepend=np.diff(x).mean()).reshape((-1,1))
        dxdx = dx ** 2
        f = np.array(self.joint_motion['pos'])
        # First derivatives:
        df = np.diff(f, axis=0, prepend=f[0].reshape((1, -1))) / dx
        cf = np.convolve(f[0][1:-1], [1, -1]) / dx
        # gf = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap') / dx
        df_norm = np.linalg.norm(df, axis=1)

        # Second derivatives:
        ddf = np.diff(f, n=2, axis=0, prepend=f[0].reshape((1, -1)), append=f[-1].reshape((1, -1))) / dxdx
        ccf = np.convolve(f[0][1:-2], [1, -2, 1]) / dxdx
        # ggf = ndimage.gaussian_filter1d(f, sigma=1, order=2, mode='wrap') / dxdx

        # self.joint_motion['vel'] = df
        self.joint_motion['acc'] = ddf

    def get_goal(self):
        return self.ref_traj

    def get_ref_curve(self):
        # curve_data = d["goal"]["curve"]
        curve_data = self.ref_traj['curve']
        curve = np.array([[c["x"], c["y"], c["z"]] for c in curve_data])
        return curve

    def plot_motion_traj(self, ts_sample, qs_sample, qds_sample=None, qdds_sample=None):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, sharex=True)
        qs_sample = np.array(qs_sample).transpose()
        for i in range(qs_sample.shape[1]):
            # plot the i-th joint trajectory
            axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
            if not qds_sample is None:
                axs[1].plot(ts_sample, np.array(qds_sample).transpose()[:, i], c="C{:d}".format(i))
            if not qdds_sample is None:
                axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        plt.show()

    def plot_finite_diffs(self):
        # import numpy as np
        from scipy import ndimage
        import matplotlib.pyplot as plt

        # Data:
        # x = np.linspace(0, 2 * np.pi, 100)
        # f = np.sin(x) + .02 * (np.random.rand(100) - .5)

        x = self.joint_motion['t']
        f = self.eef_pos.T

        # Normalization:
        # dx = x[1] - x[0]  # use np.diff(x) if x is not uniform
        # dxdx = dx ** 2

        dx = np.diff(x)
        dxdx = dx ** 2

        # First derivatives:
        df = np.diff(f) / dx
        cf = np.convolve(f[0][1:-1], [1, -1]) / dx
        # gf = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap') / dx
        df_norm = np.linalg.norm(df, axis=0)

        # Second derivatives:
        ddf = np.diff(f, 2) / dxdx[:-1]
        ccf = np.convolve(f[0][1:-2], [1, -2, 1]) / dxdx
        # ggf = ndimage.gaussian_filter1d(f, sigma=1, order=2, mode='wrap') / dxdx

        # Plotting:
        plt.figure()
        plt.plot(x, f.T[:, :3], 'k', lw=2, label='original')
        plt.plot(x[:-1], df.T, 'r.', label='np.diff, 1')
        plt.plot(x[:-1], df_norm.T, 'b-', label='speed')
        # plt.plot(x[:-2], cf[:-1], 'r--', label='np.convolve, [1,-1]')
        # plt.plot(x, gf, 'r', label='gaussian, 1')
        # plt.plot(x[:-2], ddf, 'g.', label='np.diff, 2')
        # plt.plot(x, ccf[:-2], 'g--', label='np.convolve, [1,-2,1]')
        # plt.plot(x, ggf, 'g', label='gaussian, 2')
        plt.legend()
        plt.show()
        # plt.savefig()

    def get_joint_state_to_kdl_mapping(self):
        names = self.kdl_kin.get_joint_names()  # type: list
        mapping = [self.joint_names.index(name) for name in names]
        return mapping

    def calc_eef_poses(self):
        pos = np.array(self.joint_motion['pos'])
        m = self.get_joint_state_to_kdl_mapping()
        ee_mat = np.array([self.kdl_kin.forward(q=q[m], end_link=self._eef, base_link=self._world_frame) for q in pos[:]])   # np.array((list(map(self.kdl_kin.forward, pos)))
        ee_pos = np.zeros((ee_mat.shape[0], 4))
        ee_pos[:] = ee_mat[:, :, 3]
        if self._eef == 'tool0':
            ee_pos_table = np.dot(np.linalg.inv(LogAnalyzer.w2table1), ee_pos.transpose()).transpose()
        elif self._eef == "tool_r2":
            ee_pos_table = np.dot(np.linalg.inv(LogAnalyzer.w2table2), ee_pos.transpose()).transpose()
        self.eef_pos = ee_pos_table

        # Data:
        # x = np.linspace(0, 2 * np.pi, 100)
        # f = np.sin(x) + .02 * (np.random.rand(100) - .5)

        x = self.joint_motion['t']
        f = self.eef_pos.T

        # Normalization:
        # dx = x[1] - x[0]  # use np.diff(x) if x is not uniform
        # dxdx = dx ** 2

        dx = np.diff(x)
        dxdx = dx ** 2

        # First derivatives:
        df = np.diff(f) / dx
        cf = np.convolve(f[0][1:-1], [1, -1]) / dx
        # gf = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap') / dx
        df_norm = np.linalg.norm(df, axis=0)

        # Second derivatives:
        ddf = np.diff(f, 2) / dxdx[:-1]
        ccf = np.convolve(f[0][1:-2], [1, -2, 1]) / dxdx
        # ggf = ndimage.gaussian_filter1d(f, sigma=1, order=2, mode='wrap') / dxdx

        # x = self.eef_pos[:,0]
        # y = self.eef_pos[:,1]
        # z = self.eef_pos[:,2]
        tck, u = interpolate.splprep(u=self.joint_motion['t'], x=self.eef_pos[:, :3].transpose(), k=5)
        # spline_par = interpolate.splprep(u=self.joint_motion['t'], x=self.eef_pos[:, :3].transpose(), k=5)

        pos_interp = interpolate.splev(u, tck, der=0)

        v = interpolate.splev(u, tck, der=1)
        v_abs = np.linalg.norm(v, axis=0)

        # TODO: verify that the right point is selected
        J_ee = np.array([self.kdl_kin.jacobian(q=q[m]) for q in pos[:]])  # np.array((list(map(self.kdl_kin.forward, pos)))
        vel = np.array(self.joint_motion['vel'])
        ee_vel = np.zeros((ee_mat.shape[0], 6))
        ee_vel = [np.dot(J_ee[i], vel[i, m]) for i in range(vel.shape[0])]
        if False:
            self.eef_vel = v
        elif True:
            self.eef_vel = np.array(ee_vel)


        # self.plot_motion_traj(u,pos_interp, self.eef_vel)







if __name__=='__main__':
    la = LogAnalyzer()
    la.load_data('../../test/log_file.json')
    la.calc_eef_poses()

    print(la.log)