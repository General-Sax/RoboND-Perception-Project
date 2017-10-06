#!/usr/bin/env python

import sys
import rospy
from geometry_msgs import JointState
from pr2_robot.srv import *

def at_goal(joint_position, joint_goal):
    tolerance = .05
    result = abs(pos_j1 - goal_j1) <= abs(tolerance)
    result = result and abs(pos_j2 - goal_j2) <= abs(tolerance)
    return result


def pick_place_client(test_scene_num, object_name, arm_name, pick_pose, place_pose):
    rospy.wait_for_service('pick_place_routine')
    try:
        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
        resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
        return resp
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        # return False

if __name__ == "__main__":
    joint_tracker = rospy.Subscriber('/pr2/joint_states', JointState, queue_size=1)
    print "Requesting new pick place operation..."
    pick_place_client()
