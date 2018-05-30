#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of perception exercises for the Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Brandon Kinman

import math
import random
import rospy
import rospkg
import tf

from sensor_stick.srv import GetNormals
from gazebo_msgs.srv import GetPhysicsProperties
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def capture_sample():
    '''
    Captures a PointCloud2 object using the sensor stick RGBD camera.
    
    :return: PointCloud2 - a single RGBD point cloud measurement
    '''
    get_model_state_prox = rospy.ServiceProxy('gazebo/get_model_state',GetModelState)
    model_state = get_model_state_prox('training_model','world')

    set_model_state_prox = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

    roll = random.uniform(0, 2*math.pi)
    pitch = random.uniform(0, 2*math.pi)
    yaw = random.uniform(0, 2*math.pi)
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    model_state.pose.orientation.x = quaternion[0]
    model_state.pose.orientation.y = quaternion[1]
    model_state.pose.orientation.z = quaternion[2]
    model_state.pose.orientation.w = quaternion[3]
    sms_req = SetModelStateRequest()
    sms_req.model_state.pose = model_state.pose
    sms_req.model_state.twist = model_state.twist
    sms_req.model_state.model_name = 'training_model'
    sms_req.model_state.reference_frame = 'world'
    set_model_state_prox(sms_req)

    return rospy.wait_for_message('/pr2/world/points', PointCloud2)


def initial_setup():
    '''
    Prepares the Gazebo world for generating training data.

    In particular, this routine turns off gravity, so that the objects
    spawned in front of the RGBD camera will not fall. It also deletes
    the ground plane, so that the only depth points produced will
    correspond to the object of interest (eliminating the need for
    clustering and segmentation as part of the training process)
    
    :return: None
    '''
    rospy.wait_for_service('gazebo/get_model_state')
    rospy.wait_for_service('gazebo/set_model_state')
    rospy.wait_for_service('gazebo/get_physics_properties')
    rospy.wait_for_service('gazebo/set_physics_properties')
    rospy.wait_for_service('gazebo/spawn_sdf_model')

    get_physics_properties_prox = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
    physics_properties = get_physics_properties_prox()

    physics_properties.gravity.z = 0

    set_physics_properties_prox = rospy.ServiceProxy('gazebo/set_physics_properties', SetPhysicsProperties)
    set_physics_properties_prox(physics_properties.time_step,
                                physics_properties.max_update_rate,
                                physics_properties.gravity,
                                physics_properties.ode_config)

    delete_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    try:
        delete_model_prox('ground_plane')
    except:
        pass



def spawn_model(model_name):
    '''
    Spawns a model in front of the RGBD camera.
    
    :param model_name: valid object name from {sticky_notes, book, snacks, biscuits, eraser, soap2, soap, glue}
    :return: None
    '''
    initial_pose = Pose()
    initial_pose.position.x = 0.6
    initial_pose.position.y = 0
    initial_pose.position.z = 0.85

    # Spawn the new model #
    model_path = rospkg.RosPack().get_path('pr2_robot')+'/models/'
    model_xml = ''

    with open (model_path + model_name + '/model.sdf', 'r') as xml_file:
        model_xml = xml_file.read().replace('\n', '')

    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    spawn_model_prox('training_model', model_xml, '', initial_pose, 'world')



def delete_model():
    '''
    Deletes whichever model is currently being rendered in gazebo as 'training_model'
    :return: None
    '''
    delete_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    
    # Delete the old model if it's stil around:
    delete_model_prox('training_model')
