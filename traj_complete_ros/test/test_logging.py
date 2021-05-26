#! /usr/bin/env python

import unittest
import rostest
from traj_complete_ros.LoggerProxy import LoggerProxy


class LoggerTests(unittest.TestCase):

    def setUpClass(cls):
        pass

    def test_whatever(self):
        pass

if __name__ == '__main__':
    rostest.rosrun('traj_complete_ros', 'test_logging', LoggerTests)
