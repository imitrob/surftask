#! /usr/bin/env python


import rospy
from geometry_msgs.msg import Point, Quaternion, Pose
from traject_msgs.srv import PlanTolerancedTrajecory, PlanTolerancedTrajecoryRequest, PlanTolerancedTrajecoryResponse
import numpy as np

from traj_complete_ros.toppra_eef_vel_ct import retime, plot_plan

from moveit_commander import MoveGroupCommander, RobotTrajectory


if __name__ == "__main__":
    rospy.init_node('send_path')
    rospy.sleep(0.5)
    get_traj_srv = rospy.ServiceProxy('/planTolerancedTrajecory', PlanTolerancedTrajecory)

    rospy.sleep(1.0)

    req = PlanTolerancedTrajecoryRequest()
    req.header.frame_id = 'base_link'
    
    rect_a = 0.2
    rect_b = 0.2
    res = 100
    h_start = 0.3
    h_end = 0.3

    nr_points_a = int(rect_a * res)
    nr_points_b = int(rect_b * res)
    a = np.linspace(-0.5 * rect_a, 0.5 * rect_a, nr_points_a)
    b = np.linspace(-0.5 * rect_b, 0.5 * rect_b, nr_points_b)
    h = np.linspace(h_start, h_end, 2 * (nr_points_a + nr_points_b))

    # rectangle starts in top left corner, center is in the middle
    curve_points = np.zeros(shape=(2 * (nr_points_a + nr_points_b), 3))
    curve_points[0:nr_points_a] = np.array(
        [a, nr_points_a * [rect_b * 0.5], h[0:nr_points_a]]).transpose()
    curve_points[nr_points_a:nr_points_a + nr_points_b] = np.array(
        [nr_points_b * [rect_a * 0.5], -b, h[nr_points_a:nr_points_a + nr_points_b]]).transpose()
    curve_points[nr_points_a + nr_points_b:2 * nr_points_a + nr_points_b] = np.array(
        [-1.0 * a, nr_points_a * [rect_b * (-0.5)],
         h[nr_points_a + nr_points_b:2 * nr_points_a + nr_points_b]]).transpose()
    curve_points[2 * nr_points_a + nr_points_b:] = np.array(
        [[-0.5 * rect_a] * nr_points_b, b, h[2 * nr_points_a + nr_points_b:]]).transpose()

    # remove duplicate points from data
    to_delete = np.where(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1) <= 0.0001)
    curve_points = np.delete(curve_points, to_delete, axis=0)
    # curve_normals = np.delete(curve_normals, to_delete, axis=0)


    print(curve_points)


    req.poses = [Pose(position=Point(x,y,z), orientation=Quaternion(0,0,0,1)) for x,y,z in curve_points[:]]
    req.base_to_path.translation.x = 0.4
    req.base_to_path.translation.y = 0.0
    req.base_to_path.translation.z = 0.3
    req.base_to_path.rotation.w = 1.0

    res = get_traj_srv.call(req)

    mg = MoveGroupCommander('r1_arm')

    plan = RobotTrajectory()

    plan.joint_trajectory = res.traj

    mg.go(plan.joint_trajectory.points[0].positions)
    print(plan)

    plan = retime(plan, cart_vel_limit=0.03)
    mg.execute(plan)
    rospy.sleep(2.0)


    # send goal around a circle