#!/usr/bin/env python2
"""
logging node
"""
import os

from sensor_msgs.msg import JointState
from traject_msgs.srv import LoggerControl, LoggerControlRequest, LoggerControlResponse

import rospy

from traject_msgs.msg import LoggingTask
from urdf_parser_py.urdf import URDF

import uuid

from rospy_message_converter.message_converter import convert_ros_message_to_dictionary



import json

# 0: init
# 1: running
# 2: stopped
# 3:

INIT = 0
RUNNING = 1
STOPPED = 2

class TrajectoryLogger(object):
    def __init__(self, js_topic):
        logging_folder = os.path.expanduser('~/traj_complete_log/')
        self.save_robot_xml(logging_folder)

        self.log_srv = rospy.Service('logging', LoggerControl, self.control_cb)

        self.joint_motion = {'t': [],
                             'pos': [],
                             'vel': [],
                             'acc': []}

        self.log = {'t': [],
                    'data': []}

        self.joint_names = []
        self.link_names = []

        self.ref_traj = None

        self.record = False
        self.t_start = None
        self.t_end = None
        self.state = INIT

        self.js_sub = rospy.Subscriber(js_topic, JointState, self.joint_state_cb, queue_size=100)
        self.control_srv = rospy.Service('/log_control', LoggerControl, self.control_cb)

    def save_robot_xml(self, path):
        name = uuid.uuid4().hex
        robot_xml_path = os.path.join(os.path.dirname(path), "robot_{}.xml".format(name))
        # if not os.path.exists(robot_xml_path):
        urdf_xml = rospy.get_param('robot_description')
        with open(robot_xml_path, 'w') as f:
            f.write(urdf_xml)
            # robot_urdf = URDF.from_parameter_server('robot_description')

        rospy.set_param('/robot_xml_path', os.path.basename(robot_xml_path))

    def init_vars(self):
        self.joint_motion = {'t': [],
                             'pos': [],
                             'vel': [],
                             'acc': []}

        self.log = {'t': [],
                    'data': []}

        self.joint_names = []
        self.link_names = []

        self.ref_traj = None

        self.record = False
        self.t_start = None
        self.t_end = None
        self.state = INIT

        js = rospy.wait_for_message(self.js_sub.resolved_name, JointState, timeout=rospy.Duration.from_sec(5.0))
        assert isinstance(js, JointState)
        self.joint_names = js.name

    def start(self):
        #
        rospy.loginfo('START logger node')
        self.t_start = rospy.Time.now()
        self.record = True
        self.state = RUNNING

    def stop(self):
        self.record = False
        self.t_end = rospy.Time.now()
        rospy.loginfo('STOP logger node. We recorded for {} sec.'.format((self.t_end-self.t_start).to_sec()))
        self.state = 2

    def save(self, file):
        if not os.path.exists(os.path.dirname(file)):
            os.mkdir(os.path.dirname(file))
            rospy.loginfo('create log dir: {}'.format(os.path.dirname(file)))

        with open(file, 'w') as f:
            json.dump({'joint_names': self.joint_names,'motion': self.joint_motion, 'log': self.log,
                       'goal': convert_ros_message_to_dictionary(self.ref_traj)},
                      f)
        rospy.loginfo('saved log to: {}'.format(file))

    def reset(self):
        rospy.loginfo('RESETTING logger node')
        self.record = False
        self.init_vars()

    def joint_state_cb(self, msg):
        # type: (JointState) -> None
        if not self.record:
            return
        self.joint_motion['t'].append((msg.header.stamp - self.t_start).to_sec())
        self.joint_motion['pos'].append(msg.position)
        self.joint_motion['vel'].append(msg.velocity)
        self.joint_motion['acc'].append(msg.effort)  # not working in capek simulation

    def control_cb(self, req):
        #type: (LoggerControlRequest) -> LoggerControlResponse
        # possible uses: start/stop logging
        # start/stop subscribers
        # add custom data point
        # register derived log signal as python function
        # save to file
        res = LoggerControlResponse()
        res.success = False

        if req.task.action == LoggingTask.START:
            if self.state == INIT:
                self.start()
                res.success = True
                return res
        if req.task.action == LoggingTask.SETREFERENCE:
            if self.state == INIT:
                rospy.loginfo('set path reference: {}'.format(req.task.goal.curve))
                self.ref_traj = req.task.goal
                res.success = True
                return res
        if req.task.action == LoggingTask.STOP:
            if self.state == RUNNING:
                self.stop()
                res.success = True
                return res
        if req.task.action == LoggingTask.LOG:
            if self.state == RUNNING:
                self.log_data(req.header.stamp ,req.task.data)
                res.success = True
                return res
        if req.task.action == LoggingTask.RESET:
            self.reset()
            res.success = True
            return res
        if req.task.action == LoggingTask.SAVE:
            if self.state == STOPPED:
                self.save(file=req.task.data)
                res.success = True
                return res

        return res


    # joint state -> compute eef pose and velocity, acc, jerk
    # joint pos, vel, acc, jerk
    # deviation from the curve
    #

    # def add_signal(self, name):
    #     self.signals.append(name)

    def log_data(self, stamp, data):
        rospy.loginfo('log: {}'.format(data))
        self.log['t'].append((stamp-self.t_start).to_sec())
        self.log['data'].append(data)


if __name__ == "__main__":
    rospy.init_node('logger_loewe')
    # js_topic = '/joint_states' # '/r2/joint_states'
    js_topic = '/r1/joint_states'

    log = TrajectoryLogger(js_topic=js_topic)
    rospy.spin()
