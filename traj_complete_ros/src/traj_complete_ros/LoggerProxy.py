
import rospy
from traject_msgs.msg import LoggingTask
from traject_msgs.srv import LoggerControl, LoggerControlResponse, LoggerControlRequest


class LoggerProxy(object):
    logger_srv = None

    @staticmethod
    def logger(action, data=None):
        '''

        :param action: int to select an action
                        int32 START = 1
                        int32 STOP = 2
                        int32 LOG = 3
                        int32 SAVE = 4
                        int32 RESET = 5
        :param data: string
        :return:
        '''
        if not isinstance(LoggerProxy.logger_srv, rospy.ServiceProxy):
            LoggerProxy.logger_srv = rospy.ServiceProxy('/log_control', LoggerControl, False)

        req = LoggerControlRequest()
        req.header.stamp = rospy.Time.now()
        valid_req = False
        res = None
        if action==LoggingTask.START:
            valid_req = True
            req.task.action = action
        if action==LoggingTask.SETREFERENCE:
            valid_req = True
            req.task.action = action
            req.task.goal = data
        if action == LoggingTask.STOP:
            valid_req = True
            req.task.action = action
        if action == LoggingTask.LOG:
            valid_req = True
            req.task.action = action
            req.task.data = data
        if action == LoggingTask.SAVE:
            valid_req = True
            req.task.action = action
            req.task.data = data
        if action == LoggingTask.RESET:
            valid_req = True
            req.task.action = action

        if valid_req:
            try:
                res = LoggerProxy.logger_srv.call(req)
            except rospy.ServiceException as e:
                rospy.logwarn_once('LoggerNode is not running. Will log to rosout.')
                rospy.loginfo(req)
                return True
            except AttributeError as e:
                rospy.logerr(e)
                raise e

        assert isinstance(res, LoggerControlResponse)
        return res.success



