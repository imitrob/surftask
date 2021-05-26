#!/usr/bin/env python2
"""
ROS action server example
"""
import numpy as np
import rospy
import actionlib
from traject_msgs.msg import CurveExecutionAction, CurveExecutionGoal, CurveExecutionResult, CurveExecutionFeedback


def execute(goal):
    curve = np.array([[c.x, c.y, c.z] for c in goal.curve])
    normals = np.array([[n.x, n.y, n.z] for n in goal.normals])
    pause = rospy.Duration(0.005)
    n_points = curve.shape[0]
    success = True
    rospy.loginfo("Starting curve execution.")
    for i, (c, n) in enumerate(zip(curve, normals)):
        if server.is_preempt_requested():
            server.set_preempted()
            success = False
            break

        rospy.loginfo("> c: {}\t n: {} ({:.2f}%)".format(c, n, float(i) / n_points * 100))
        server.publish_feedback(CurveExecutionFeedback(progress=float(i) / n_points * 100))
        rospy.sleep(pause)
    if success:
        # create result/response message
        server.set_succeeded(CurveExecutionResult(True))
        rospy.loginfo('Action successfully completed')
    else:
        server.set_aborted(CurveExecutionResult(False))
        rospy.loginfo('Whoops')


if __name__ == '__main__':
    rospy.init_node('dummy_controller')
    # Similarly to service, advertise the action server
    server = actionlib.SimpleActionServer('curve_executor', CurveExecutionAction, execute, False)
    server.start()
    rospy.spin()
