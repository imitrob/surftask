#!/usr/bin/env python2
"""
ROS action server example
"""
import copy

import numpy as np
from relaxed_ik.msg import EEPoseGoals, JointAngles
from scipy.signal import argrelextrema
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header

import rospy
import actionlib
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from moveit_commander.move_group import MoveGroupCommander
from moveit_msgs.msg import RobotTrajectory, RobotState
from scipy import interpolate
from scipy.spatial.distance import euclidean
from tf.transformations import translation_matrix, quaternion_from_matrix, translation_from_matrix, quaternion_matrix
from traject_msgs.msg import CurveExecutionAction, CurveExecutionGoal, CurveExecutionResult, CurveExecutionFeedback, \
    ExecutionOptions, LoggingTask, PointVel
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

from RelaxedIK.relaxedIK import RelaxedIK
# from start_here import config_file_name
from traj_complete_ros.LoggerProxy import LoggerProxy
from traj_complete_ros.geometry import R_axis_angle
from traj_complete_ros.trajectory_action_client import TrajectoryActionClient
from control_msgs.msg import FollowJointTrajectoryActionFeedback, FollowJointTrajectoryActionGoal, \
    FollowJointTrajectoryGoal, FollowJointTrajectoryFeedback

from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF, Collision, Mesh, Cylinder, Box, Sphere

from franka_interface import ArmInterface


def get_current_robot_state(mg):
    joints = mg.get_active_joints()
    js = mg.get_current_joint_values()
    if np.allclose(js, [0.0 for _ in range(7)]):
        js = mg.get_current_joint_values()

    rs1 = RobotState()
    rs1.joint_state.name = joints
    rs1.joint_state.position = js

    return rs1


def state_from_plan_point(plan, point):
    rs = RobotState()
    rs.joint_state.name = plan.joint_trajectory.joint_names
    pt = plan.joint_trajectory.points[point]
    assert isinstance(pt, JointTrajectoryPoint)
    rs.joint_state.position = pt.positions
    rs.joint_state.velocity = pt.velocities

    return rs

class CurveExecutor(object):
    def __init__(self):
        self.server = actionlib.SimpleActionServer('curve_executor', CurveExecutionAction, self.execute, False)

        self.arm = ArmInterface()

        self.arm.move_to_neutral()

        self.arm.endpoint_effort()

        self.pub = rospy.Publisher('/motion_goal', PointVel, queue_size=1)

        # load kdl
        eef = 'panda_rightfinger'
        self._world_frame = 'world'
        self.robot_urdf = URDF.from_parameter_server('robot_description')
        self.kdl_kin = KDLKinematics(self.robot_urdf, self._world_frame, eef)

        ####################################################################################################################
        # self.relaxedIK = RelaxedIK.init_from_config(config_file_name)

        # prepare JointTrajectory for relaxedik callback
        # self.traj = JointTrajectory()
        # self.traj.joint_names = self.mg.get_active_joints()
        # self.traj.points.append(JointTrajectoryPoint())

        self.eef_speed = 0.1 # m/s


        self.last_js = JointState()
        self.las_js_diff = 10

        self.js_sub = rospy.Subscriber('/r2/joint_states', JointState, callback=self.js_cb, queue_size=1)

        self.server.start()

    def switch_relaxed_ik(self, on=None):
        if on is None:
            # toggle mode
            on = not self.relaxedik_clutch

        if self.relaxedik_clutch == on:
            rospy.loginfo('relaxed ik clutch is {}, as it should be. Doing noting.'.format(self.relaxedik_clutch))
            return
        #send robot home
        if not self.relaxedik_clutch:
            rospy.loginfo('relaxed ik clutch is disengaged. Preparing for engagement.')
            js_lpos = self.mg.get_named_target_values('L_position')
            self.mg.go(js_lpos)
        else:
            rospy.loginfo('relaxed ik clutch is engaged. Preparing for DISengagement.')
            goal = EEPoseGoals()
            goal.ee_poses.append(Pose(orientation=Quaternion(w=1)))
            self.relaxedik_pub.publish(goal)
            rospy.sleep(3.0)

        if on:
            goal = EEPoseGoals()
            goal.ee_poses.append(Pose(orientation=Quaternion(w=1)))
            self.relaxedik_pub.publish(goal)
            rospy.sleep(0.5)
            rospy.loginfo('engaging relaxed ik clutch.')
            self.relaxedik_clutch = True
        else:
            rospy.loginfo('disengaging relaxed ik clutch.')
            self.relaxedik_clutch = False

    def get_eef_pose(self):
        return self.arm.endpoint_pose()


    def js_cb(self, msg):
        #type: (JointState) -> None
        # assert isinstance(msg, JointState)
        # if 'r2' in msg.name[0]:
        self.last_js = msg
        self.las_js_diff = np.linalg.norm(np.array(msg.position) - self.relaxed_ik_js)
        # print("received js: {}".format(msg.position))
        # else:
        #     return

    def rik_cb(self, msg):
        assert isinstance(msg, JointAngles)
        # msg.angles.data
        # trajectory_msgs / JointTrajectory

        if not self.relaxedik_clutch:
            return

        if np.any(np.isnan(msg.angles.data)):
            rospy.logwarn('no relaxed ik solution.')
            return

        pt = self.traj.points[0]  # type: JointTrajectoryPoint
        pt.positions = msg.angles.data

        self.relaxed_ik_js = msg.angles.data

        # cur_pose = self.mg.get_current_pose(self.mg.get_end_effector_link())
        # cur_point = np.array([cur_pose])
        cur_js = self.mg.get_current_joint_values()
        cur_pose_mat = self.kdl_kin.forward(q=cur_js, base_link='world',
                                            end_link=self.mg.get_end_effector_link())
        cur_point = translation_from_matrix(cur_pose_mat)

        new_pose_mat = self.kdl_kin.forward(q=msg.angles.data, base_link='world', end_link=self.mg.get_end_effector_link())
        new_point = translation_from_matrix(new_pose_mat)

        dist = euclidean(new_point, cur_point)

        pt.time_from_start = rospy.Duration.from_sec(dist/self.eef_speed)

        self.traj.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.05)
        self.traj_cmd.publish(self.traj)


    def get_joint_limits(self, joints):
        lower = []
        upper = []
        vel = []
        effort = []
        for joint in joints:
            lower += [self.robot_urdf.joint_map[joint].limit.lower]
            upper += [self.robot_urdf.joint_map[joint].limit.upper]
            vel += [self.robot_urdf.joint_map[joint].limit.velocity]
            effort += [self.robot_urdf.joint_map[joint].limit.effort]
        return lower, upper, vel, effort

    def retime(self, plan):
        import toppra as ta
        assert isinstance(plan, RobotTrajectory)
        lower, upper, vel, effort = self.get_joint_limits(self.mg.get_active_joints())

        alims = len(lower)* [1.0]

        ss = [pt.time_from_start.to_sec()*0.01 for pt in plan.joint_trajectory.points]
        way_pts = [list(pt.positions) for pt in plan.joint_trajectory.points]

        path = ta.SplineInterpolator(ss, way_pts)

        pc_vel = ta.constraint.JointVelocityConstraint(np.array([lower, upper]).transpose())

        pc_acc = ta.constraint.JointAccelerationConstraint(np.array(alims))

        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc], path)
        # print(instance)
        # instance2 = ta.algorithm.TOPPRAsd([pc_vel, pc_acc], path)
        # instance2.set_desired_duration(60)
        jnt_traj = instance.compute_trajectory()

        # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
        ts_sample = np.linspace(0, jnt_traj.duration, np.ceil(100*jnt_traj.duration))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)

        new_plan = copy.deepcopy(plan)
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


    def traj_feedback_cb(self, msg):
        goal = self.traj_client._goal
        if not isinstance(goal, FollowJointTrajectoryGoal):
            return
        assert isinstance(msg, FollowJointTrajectoryFeedback)

        points = self.get_traj_pts(goal)
        desired = np.array(msg.desired.positions)

        dist = (np.linalg.norm(points-desired, axis=1))
        idx = np.argmin(dist)

        fb = CurveExecutionFeedback()
        fb.progress = float(idx)/ points.shape[0]
        self.server.publish_feedback(fb)
        # TODO: the feedback could be misleading in the case that the trajectory is crossing over itself.
        return

    def normalize_vectors(self, y_new):
        scale = 1 / np.linalg.norm(y_new, axis=1)
        y_norm = np.array([y_new[:, 0] * scale, y_new[:, 1] * scale, y_new[:, 2] * scale]).transpose()
        return y_norm

    @staticmethod
    def plot_spline(tck_param, start=0.0, end=1.0, points=[]):
        from mpl_toolkits import mplot3d
        # %matplotlib inline
        # import numpy as np
        import matplotlib.pyplot as plt

        # spline = interpolate.BSpline(tck_param[0], tck_param[1], tck_param[2], extrapolate=False)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        u = np.linspace(start, end, 300)
        x = interpolate.splev(u, tck=tck_param)

        ax.plot3D(x[0], x[1], x[2], 'gray')
        # ax.scatter(x[0], x[1], x[2])

        if len(points) > 0:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro')

            # ax.scatter(points[:,0], points[:,1], points[:,2], marker='x')

        fig.show()

    def execute(self, goal):
        assert isinstance(goal, CurveExecutionGoal)
        success = False


        self.eef_speed = goal.opt.cart_vel_limit

        exec_engine = goal.opt.executor_engine
        # self.switch_relaxed_ik(on=(exec_engine==ExecutionOptions.RELAXEDIK))

        # prepare trajectory
        curve = np.array([[c.x, c.y, c.z, 1.0] for c in goal.curve])
        normals = np.array([[n.x, n.y, n.z] for n in goal.normals])

        # remove duplicate points from data
        to_delete = np.where(np.linalg.norm(curve[1:] - curve[:-1], axis=1) <= 0.0001)
        curve = np.delete(curve, to_delete, axis=0)
        normals = np.delete(normals, to_delete, axis=0)


        if goal.header.frame_id != "table2":
            if goal.header.frame_id == "":
                # post processing for curves without camera calibration
                curve = curve - curve.mean(axis=0) + np.array([0.0, 0.0, 0.0, 1.0])
                # curve = curve - curve.mean(axis=0) + np.array([0.1, 0.0, 0.2, 1.0])


                # v1*v2 = | v1 | | v2 | cos(angle)
                normal_avg = np.mean(normals, axis=0)
                v2 = np.array([0,0,1])
                angle = np.arccos(np.dot(v2, normal_avg))  # assumes vectors to be normalized!!!

                axis = np.cross(v2, normal_avg)

                R = np.eye(3)
                R_axis_angle(R, axis=axis, angle=angle)

                shift = np.eye(4)
                shift[:3, :3] = R
                curve_new = np.dot(curve, shift)
                normals_new = np.dot( normals, R)

                curve = curve_new
                normals = normals_new

                curve = curve + np.array([0.1, 0.0, 0.2, 0.0])
            else:
                # transform points according to frame
                raise NotImplementedError
        else:
            pass


        # points are relative to the frame '/table2' and relaxedIK requires the points relative to the reference pose, which
        # is the robot standing straight up: world to r2_joint_tip pos: -0.0019918; -0.85177; 1.2858      quat: -0.0011798; -0.00090396; -0.0031778; 0.99999
        # world to table2: 0.42783; -0.85741; -0.0054846    quat: -0.0011798; -0.0008628; -0.0030205; 0.99999
        # new world2ref
        # 0.39933; -0.85591; 0.58248
        # quat: 0.0030643; 0.99999; -0.0012084; 0.00084322
        # TODO: relaxed IK reference frame changed

        ########################### PANDA transforms ########

        w2table2 = translation_matrix(np.array([0.3, 0.0, 0.0]))
        table22world = np.linalg.inv(w2table2)

        curve_tf = np.dot(w2table2, curve.transpose()).transpose()[:, :3]

        ########################### NEW ####################
        # w2ref = translation_matrix(np.array([0.39933, -0.85591, 0.58248]))
        # w2ref_rot = quaternion_matrix([0.0030643, 0.99999, -0.0012084, 0.00084322])
        # ref2w_rot = np.linalg.inv(w2ref_rot)
        # # w2ref[:3,:3] = w2ref_rot[:3,:3]
        # # w2ref = translation_matrix(np.array([-0.0019918, -0.85177, 1.2858]))
        # ref2w = np.linalg.inv(w2ref)
        # w2table2 = translation_matrix(np.array([0.42783, -0.85741, -0.0054846]))
        #
        # ref2table = np.dot(ref2w, w2table2)
        #
        # curve_tf = np.dot(ref2table, curve.transpose()).transpose()[:, :3]

        ############## NEW END ###################

        # w2ref = translation_matrix(np.array([0.39933, -0.85591, 0.58248]))
        # w2ref_rot = quaternion_matrix([0.0030643, 0.99999, -0.0012084, 0.00084322])
        # # w2ref[:3,:3] = w2ref_rot[:3,:3]
        # # w2ref = translation_matrix(np.array([-0.0019918, -0.85177, 1.2858]))
        # ref2w = np.linalg.inv(w2ref)
        # w2table2 = translation_matrix(np.array([0.42783, -0.85741, -0.0054846]))
        #
        # transform = np.dot(ref2w, w2table2)
        #
        # curve_tf = np.dot(transform, curve.transpose()).transpose()[:,:3]

        ############# OLD end ############

        # tck_params, xy_as_u = interpolate.splprep(curve_tf.transpose(), k=5, s=0.01, per=0)
        x = curve_tf[:, 0]
        y = curve_tf[:, 1]
        z = curve_tf[:, 2]
        tck_params, xy_as_u = interpolate.splprep([x, y, z], k=5, s=0, per=0)

        # CurveExecutor.plot_spline(tck_params, points=curve_tf)
        pt_dot = np.transpose(np.array(interpolate.splev(xy_as_u, tck_params, der=1)))
        # y_new = np.cross(pt_dot, -normals)
        y_new = np.cross(-normals, pt_dot)
        #TODO: what if the normals and the path direction are not orthogonal?

        scale = 1/np.linalg.norm(y_new,axis=1)
        y_norm = np.array([y_new[:,0] * scale, y_new[:,1] * scale, y_new[:,2] * scale]).transpose()
        x_norm = self.normalize_vectors(pt_dot)
        z_norm = - self.normalize_vectors(normals)

        # M = np.zeros((len(pt_dot),4,4))
        # M[:,0,0:3] = x_norm
        # M[:,1,0:3] = y_norm
        # M[:,2,0:3] = z_norm
        # M[:, 3, 3] = 1

        M = np.zeros((len(pt_dot), 4, 4))
        M[:, 0:3, 0] = x_norm
        M[:, 0:3, 1] = y_norm
        M[:, 0:3, 2] = z_norm
        M[:, 3, 3] = 1

        # quats = quaternion_from_matrix(M[0])

        # M_relIK = np.matmul(ref2w_rot, M)

        # quat_relaxed_ik = np.array(list(map(quaternion_from_matrix, M_relIK[:])))

        quats_path = np.array(list(map(quaternion_from_matrix, M[:])))

        # quats = np.array([[0,1,0,0] for i in range(M.shape[0])])
        quats = np.zeros(shape=quats_path.shape)
        quats[:,1] = 1.0

        if goal.opt.tool_orientation == ExecutionOptions.USE_TOOL_ORIENTATION:
            quats = quats_path

        if rospy.get_param('make_RGB_HEDGEHOG', default=False):   # 'make_RGB_HEDGEHOG':
            pub1 = rospy.Publisher('/my_Marker', Marker, queue_size=50)

            def pub_arrow(point, direction, idx, ns, color=(1,0,0,1)):
                m = Marker()
                m.header.frame_id = 'table2'
                m.ns = ns
                m.id = idx
                m.type = m.ARROW
                m.action = m.ADD
                m.pose.orientation.w = 1  # to avoid quaternion not initialized warning in rviz
                m.points.append(Point(*point))
                m.points.append(Point(*(point+direction *0.03)))
                m.scale.x = 0.005
                m.scale.y = 0.01

                m.color.r = color[0]
                m.color.g = color[1]
                m.color.b = color[2]
                m.color.a = color[3]

                pub1.publish(m)

            for idx, (point, direct) in enumerate(zip(curve[:], x_norm[:])):
                pub_arrow(point[:3], direct, idx=idx, ns='x', color=[1,0,0,0.3])
                rospy.sleep(0.01)

            for idx, (point, direct) in enumerate(zip(curve[:], y_norm[:])):
                pub_arrow(point[:3], direct, idx=idx, ns='y',color=[0,1,0,0.3])
                rospy.sleep(0.01)
            for idx, (point, direct) in enumerate(zip(curve[:], z_norm[:])):
                pub_arrow(point[:3], direct, idx=idx, ns='z', color=[0,0,1,0.3])
                rospy.sleep(0.01)

        pub2 = rospy.Publisher('/ee_pose', PoseStamped, queue_size=50)

        def pub_pose(point, quat):
            pose = PoseStamped()
            pose.header.frame_id = 'table2'
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            pub2.publish(pose)

        if False:
            for pos, quat in zip(curve[:], quats):
                pub_pose(pos, quat)
                rospy.sleep(0.01)

        #change of basis
        # M = axis of new coordinate system normalized ()

        # https://stackoverflow.com/questions/49635428/tangent-to-curve-interpolated-from-discrete-data


        # move to traj start

        def send_pose_goal(point, ori, vel, rot):
            pvel = PointVel()
            pvel.ang_vel_xyz = rot
            pvel.lin_vel = vel

            pvel.pose.position = Point(*point)
            pvel.pose.orientation = Quaternion(*ori)

            self.pub.publish(pvel)

        def integrate_piecewise(boundaries, tck):
            fst = [interpolate.splint(a=a, b=b, tck=tck, full_output=0) for a, b in
                   zip(boundaries[:-1], boundaries[1:])]
            return np.linalg.norm(fst, axis=1)

        res = integrate_piecewise(xy_as_u, tck=tck_params)
        res = np.insert(res, 0, 0.0, axis=0)
        dist_fn = np.cumsum(res)

        dist2par = interpolate.interp1d(dist_fn, xy_as_u)
        par2dist = interpolate.interp1d(xy_as_u, dist_fn)
        par2vel = interpolate.interp1d(xy_as_u, pt_dot, axis=0)


        send_pose_goal(curve_tf[0], [0,1,0,0], np.zeros(3), np.zeros(3))
        rospy.sleep(3.0)

        print('at starting pose.')

        draw = self.draw_path()

        rospy.sleep(1.0)
        rate = rospy.Rate(100)
        LoggerProxy.logger(action=LoggingTask.LOG, data='start')

        # for idx, (pos, quat) in enumerate(zip(curve_tf, quats)):
        par = 0.0
        dist = 0.0
        dist_max = dist_fn[-1]

        dt = 0.010

        def findPar(pos, vel, last_par):
            par = np.linspace(np.max([0.0, last_par - 0.005]), np.min([1.0, last_par + 0.005]), 300)
            dists = np.linalg.norm(interpolate.splev(par, tck_params) - pos.reshape(3, -1), axis=0)
            vels = interpolate.splev(par, tck_params, der=1)
            vels = np.array(vels)
            vel = vel.reshape((3, 1))
            # res = (np.array(vels) * vel)

            # part = vels[:,0]
            res_test = np.array([np.dot(vels[:, i], vel) for i in range(vels.shape[-1])])

            res2 = np.linalg.norm(vels, axis=0).reshape((-1, 1))
            res3 = np.linalg.norm(vel)
            cos_vel_angles = res_test / (res2 * res3)

            # mins_dist = argrelextrema(dists, np.less, axis=0)
            # mins_angles = argrelextrema(np.abs(cos_vel_angles), np.less, axis=0)
            idx = np.argmin(100 * dists.reshape((-1,1)) + res3/0.05 * np.abs(cos_vel_angles))
            return max(par[idx], last_par+0.00001)


        while par < 1.0:
            point = self.arm.endpoint_pose()['position']
            lin_vel = self.arm.endpoint_velocity()['linear']
            par = min(findPar(point, lin_vel, last_par=par), 1.0)
            par = max(dist2par(par2dist(par) + dt *self.eef_speed), par)

            # print(par)
            # dist += dt*self.eef_speed
            # par = dist2par(min(dist,dist_max))

            pos = interpolate.splev(par, tck_params)
            vel = par2vel(par)
            vel = vel/np.linalg.norm(vel) * self.eef_speed
            rot = np.zeros(3)

            send_pose_goal(pos, [0,1,0,0], vel, rot)

            draw.next()
            if self.server.is_preempt_requested():
                self.server.set_preempted()
                success = False
                send_pose_goal(pos, [0,1,0,0], np.zeros(3), np.zeros(3))
                break
            else:
                success = True

            rate.sleep()
            # rospy.sleep(0.1)

        draw.close()


        # destroy the painting iterator

        rospy.sleep(1.0)
        LoggerProxy.logger(action=LoggingTask.LOG, data='end')

        send_pose_goal([0.4, 0, 0.45], [0, 1, 0, 0], np.zeros(3), np.zeros(3))
        rospy.sleep(3.0)


        if success:
            # create result/response message
            self.server.set_succeeded(CurveExecutionResult(True))
            rospy.loginfo('Action successfully completed')
        else:
            self.server.set_aborted(CurveExecutionResult(False))
            rospy.loginfo('Whoops')

        return





    def draw_path(self, mg=None, size=0.002, color=ColorRGBA(1, 0, 0, 1), ns='path'):
        publisher = rospy.Publisher('/ee_path', Marker, queue_size=1)
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = rospy.Time.now()
        m.type = m.SPHERE_LIST
        m.pose.orientation.w = 1
        m.scale.x = size
        m.scale.y = size
        m.scale.z = size
        m.color = color
        # m.color.a = 0.9
        # m.color.r = 1.0
        m.action = m.ADD
        m.ns = ns
        m.pose.position.x = 0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.x
        m.pose.position.y = 0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.y
        m.pose.position.z = -0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.z

        # while ta_client.state() != GoalStatus.SUCCEEDED:
        pose = PoseStamped(header=Header(frame_id='world', stamp=rospy.Time.now()))
        while True:
        # draw the line
            pose.header.stamp = rospy.Time.now()
            # pose = mg.get_current_pose(mg.get_end_effector_link())  # type: PoseStamped
            pose_dict = self.get_eef_pose()
            # joint_angles = ta_client.get_current_tp().positions
            # fk_result = fk.getFK(mg.get_end_effector_link(), mg.get_active_joints(), joint_angles)  # type: GetPositionFKResponse
            p = Point(*pose_dict['position'])
            # p.x = pose.pose.position.x
            # p.y = pose.pose.position.y
            # p.z = pose.pose.position.z

            m.colors.append(ColorRGBA(m.color.r, m.color.g, m.color.b, m.color.a))
            m.action = Marker.ADD

            m.points.append(p)

            publisher.publish(m)
            yield True

    def get_traj_pts(self, traj_goal):
        if self.traj_goal_pt is None:
            self.traj_goal_pt = np.array([pt.positions for pt in traj_goal.trajectory.points])
        return self.traj_goal_pt



if __name__ == '__main__':
    rospy.init_node('dummy_controller')

    ce = CurveExecutor()
    # Similarly to service, advertise the action server
    # server = actionlib.SimpleActionServer('curve_executor', CurveExecutionAction, execute, False)
    # server.start()
    rospy.spin()
