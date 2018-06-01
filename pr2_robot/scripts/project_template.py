#!/usr/bin/env python

# Import modules
import sys
import glob
import time
import pickle

import yaml
import rospy
import tf
# import pcl # <- was previously arriving via 'from sensor_stick.pcl_helper import *', not currently working.
import sklearn
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter



save_dir = './savefiles/'
save_clouds_to_pcd = True


class PCDSaver:
    def __init__(self, output_directory, is_active):
        self.is_active = is_active
        self.output_directory = output_directory
        self.pcd_list_on_init = glob.glob(output_directory + '*.pcd')
        self.pcd_list = self.pcd_list_on_init[:]
        self.pcd_files_written = []
        self.requests_not_fulfilled = []

    def dump(self, pcl_cloud, filename):
        if not self.is_active:
            pass
        elif filename not in self.pcd_list:
            pcl.save(pcl_cloud, self.output_directory + filename)
            self.pcd_list.append(filename)
            self.pcd_files_written.append(filename)
        else:
            self.requests_not_fulfilled.append(filename)

saver = PCDSaver(save_dir, save_clouds_to_pcd)
save_pcd = saver.dump

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


def stat_outlier_filter(pcl_cloud, mean_k=100, threshold_scale=0.5):
    '''

    :param pcl_cloud:
    :param mean_k:
    :param threshold_scale:
    :return:
    '''
    # Much like the previous filters, we start by creating a filter object:
    outlier_filter = pcl_cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(mean_k)
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(threshold_scale)
    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
    return cloud_filtered


def voxel_downsample(pcl_cloud, leaf_scale, coeffs=(1.0, 1.0, 1.0)):
    '''

    :param pcl_cloud: pcl representation of point cloud to downsample
    :param leaf_scale: sets default scale for voxel edge, units of meters
    :param coeffs: linear scaling factors for edges x, y, z respectively
    :return: downsampled point cloud
    '''
    vox = pcl_cloud.make_voxel_grid_filter()
    x_leaf, y_leaf, z_leaf = (leaf_scale * coeff for coeff in coeffs)
    vox.set_leaf_size(x_leaf, y_leaf, z_leaf)
    cloud_filtered = vox.filter()
    return cloud_filtered

def passthrough_filter(pcl_cloud, axis_min=0.7, axis_max=1.2, filter_axis='z'):
    passthrough = pcl_cloud.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    # Recommended/Example settings: min: 0.6, max: 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()
    return cloud_filtered

def ransac_extract(pcl_cloud, max_distance=0.01, model=pcl.SACMODEL_PLANE, method=pcl.SAC_RANSAC):
    seg = pcl_cloud.make_segmenter()
    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    # Extract inliers
    extracted_inliers = pcl_cloud.extract(inliers, negative=False)
    # Extract outliers
    extracted_outliers = pcl_cloud.extract(inliers, negative=True)
    return extracted_inliers, extracted_outliers

def cluster_indices(dark_cloud, clusterTolerance=0.01, minClusterSize=40, maxClusterSize=500):
    '''
    
    :param dark_cloud: 
    :param clusterTolerance: 
    :param minClusterSize: 
    :param maxClusterSize: 
    :return: 
    '''
    # construct a k-d tree from the cloud_objects point cloud
    tree = dark_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = dark_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(clusterTolerance)
    ec.set_MinClusterSize(minClusterSize)
    ec.set_MaxClusterSize(maxClusterSize)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    obj_cluster_indices = ec.Extract()
    return obj_cluster_indices

def one_euclidean_cloud_from_indices(dark_cloud, obj_cluster_indices_list):
    '''
    
    :param dark_cloud: 
    :param obj_cluster_indices_list: 
    :return: 
    '''
    cluster_color = get_color_list(len(obj_cluster_indices_list))
    color_cluster_point_list = []
    # Assign each point a color based on its cluster affiliation and append to
    # unified list of colored points
    for j, indices in enumerate(obj_cluster_indices_list):
        for indice in indices:
            color_cluster_point_list.append([dark_cloud[indice][0],
                                            dark_cloud[indice][1],
                                            dark_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])
    # Create new cloud containing all clusters,
    # with each cluster's points now uniquely colored
    pcl_clustered_cloud = pcl.PointCloud_PointXYZRGB()
    pcl_clustered_cloud.from_list(color_cluster_point_list)
    return pcl_clustered_cloud

def object_clouds_from_indices(pcl_cloud, obj_cluster_indices_list, negative_toggle=False):
    '''
    
    :param pcl_cloud: 
    :param obj_cluster_indices_list: 
    :param negative_toggle: 
    :return: 
    '''
    # Assign each point a color based on its cluster affiliation and append to
    # unified list of colored points
    cluster_list = []
    pcl_cloud_list = []
    for j, indices in enumerate(obj_cluster_indices_list):
        pcl_cloud_list.append(pcl_cloud.extract(indices, negative=negative_toggle))
        # for indice in indices:
        #     cluster_list[-1].append([pcl_cloud[indice][0],
        #                             pcl_cloud[indice][1],
        #                             pcl_cloud[indice][2],
        #                             pcl_cloud[indice][3]])
        # pcl_cloud_gen = pcl.PointCloud_PointXYZRGB()
        # pcl_cloud_gen.from_list(cluster_list[-1])
        # pcl_cloud_list.append(pcl_cloud_gen)
    return pcl_cloud_list


# def save_pcd(pcl_cloud, filename):
#     '''
#
#     :param pcl_cloud:
#     :param filename:
#     :return:
#     '''
#     global pcd_list
#     if filename not in pcd_list:
#         pcd_list.append(filename)
#         pcl.save(pcl_cloud, save_dir+filename)
#     else:
#         pass

def centroid_of_(object_name, pcl_cloud=None, cast_python_float=True):
    '''
    
    :param object_name: 
    :param pcl_cloud: 
    :param cast_python_float: 
    :return: 
    '''
    global tracked_centroids
    if pcl_cloud:
        tracked_centroids[object_name] = np.mean(pcl_cloud, axis=0)[:3]

    if cast_python_float:
        return [np.asscalar(tracked_centroids[object_name][i]) for i in (0, 1, 2)]

    return tracked_centroids[object_name]


def collision_cloud(target_item_name, pcl_cloud):
    '''
    :param target_item_name: 
    :param pcl_cloud: 
    :return: 
    '''
    # pcl_cloud here should be from before the ransac segmentation step
    global object_list_param, completed_transports, object_ref_dict
    collidable = pcl_cloud
    # for all objects loaded into scene
    for item in object_list_param:
        # if an item has already been pick-placed or if it is our target...
        if item in completed_transports or item == target_item_name:
            # then exclude it from collision model by extracting its indices
            collidable = collidable.extract(object_ref_dict[item]['indices'], negative=True)
        else:
            pass
    save_pcd(collidable, '07_collision_cloud.pcd')
    ros_collision_cloud = pcl_to_ros(collidable)
    return ros_collision_cloud


def joint_state_digester(joint_state_data, abs_ang_delta_tolerance=0.01):
    '''
    
    :param joint_state_data: 
    :param abs_ang_delta_tolerance: 
    :return: 
    '''
    global world_joint_pos, world_joint_moving

    angle_delta = joint_state_data.position[-1] - world_joint_pos
    world_joint_pos = joint_state_data.position[-1]
    world_joint_moving = (np.absolute(angle_delta) > abs_ang_delta_tolerance)

    print("\nworld joint delta: "+str(angle_delta))
    print("considered moving: "+str(world_joint_moving))

# bot pivot manager
def pivot_bot(world_joint_goal, absolute_angle_tolerance, auto_recenter=True):
    '''
    
    :param world_joint_goal: 
    :param absolute_angle_tolerance: 
    :param auto_recenter: 
    :return: 
    '''
    global world_joint_pos, world_joint_moving, l_pivoted, r_pivoted
    sgn = np.sign(world_joint_goal)
    output_text = ''
    task_complete = False

    if world_joint_moving:
        # return 'joint in motion...'
        return
    else:
        pass

    if sgn == -1: # If pivoting right:
        if r_pivoted:
            task_complete = True
        elif not world_joint_pos <= ((world_joint_goal) + absolute_angle_tolerance):
            world_joint_pub.publish(world_joint_goal)
            output_text += 'pivoting right to {}; '.format(round(world_joint_goal, 5))
        else:
            r_pivoted = True
            task_complete = True
            output_text += 'pivot right complete! '
    elif sgn == 1: # If pivoting left:
        if l_pivoted:
            task_complete = True
        elif not world_joint_pos >= ((world_joint_goal) - absolute_angle_tolerance):
            world_joint_pub.publish(world_joint_goal)
            output_text += 'pivoting left to {}; '.format(round(world_joint_goal, 5))
        else:
            l_pivoted = True
            task_complete = True
            output_text += 'pivot left complete! '
    else:
        pass
    if (auto_recenter and task_complete) or sgn == 0:
        if np.fabs(world_joint_pos) > absolute_angle_tolerance:
            world_joint_pub.publish(0.0)
            if auto_recenter:
                output_text += 'automatically re-centering robot.'
            else:
                output_text += 'pivoting to centered position...'

    elif np.fabs(world_joint_pos) < absolute_angle_tolerance:
        output_text += 'beginning pivot!'
    else:
        output_text += 'halted at target orientation away from center.'

    print(output_text)
    return

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    '''
    
    :param test_scene_num: 
    :param arm_name: 
    :param object_name: 
    :param pick_pose: 
    :param place_pose: 
    :return: 
    '''
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    '''
    
    :param yaml_filename: 
    :param dict_list: 
    :return: 
    '''
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    '''
    
    :param pcl_msg: 
    :return: 
    '''
    ### Exercise-2 TODOs:
    ### Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)
    save_pcd(pcl_cloud, '00_raw_cloud.pcd')

    ### PassThrough Filter
    Z_AXIS_MIN = 0.4
    Z_AXIS_MAX = 0.95
    Y_AXIS_MIN = -0.4
    Y_AXIS_MAX = 0.4
    X_AXIS_MIN = 0.35
    X_AXIS_MAX = 0.9
    # passthrough filter out the approximate height of the table surface and objects
    pcl_cloud = passthrough_filter(pcl_cloud, axis_min=Z_AXIS_MIN, axis_max=Z_AXIS_MAX, filter_axis='z')
    # passthrough filter out the xy-range which corresponds to the table
    pcl_cloud = passthrough_filter(pcl_cloud, axis_min=Y_AXIS_MIN, axis_max=Y_AXIS_MAX, filter_axis='y')
    pcl_cloud = passthrough_filter(pcl_cloud, axis_min=X_AXIS_MIN, axis_max=X_AXIS_MAX, filter_axis='x')
    save_pcd(pcl_cloud, '01_passthrough_filtered.pcd')

    ### Statistical Outlier Filtering
    MEAN_K_1 = 10
    THRESHOLD_SCALE_1 = 0.8

    pcl_cloud = stat_outlier_filter(pcl_cloud, mean_k=MEAN_K_1, threshold_scale=THRESHOLD_SCALE_1)
    save_pcd(pcl_cloud, '02a_outlier_filtered.pcd')

    MEAN_K_2 = 10
    THRESHOLD_SCALE_2 = 1.6

    pcl_cloud = stat_outlier_filter(pcl_cloud, mean_k=MEAN_K_2, threshold_scale=THRESHOLD_SCALE_2)
    save_pcd(pcl_cloud, '02b_outlier_refiltered.pcd')


    ### Voxel Grid Downsampling
    # Choose a voxel (also known as leaf) size
    LEAF_SCALE = 0.005 # meters per voxel_edge - this will affect the required level for MIN_CLUSTER_SIZE
    XYZ_VOXEL_COEFFS=(1.0, 1.0, 1.0) # scale xyz dimensions of voxel independently
    pcl_cloud = voxel_downsample(pcl_cloud, leaf_scale=LEAF_SCALE, coeffs=XYZ_VOXEL_COEFFS)
    save_pcd(pcl_cloud, '03_voxel_downsampled.pcd')

    ### RANSAC Plane Segmentation
    MAX_DIST = 0.0025
    pcl_cloud_table, pcl_cloud_objects = ransac_extract(pcl_cloud, max_distance=MAX_DIST)

    save_pcd(pcl_cloud_table, '04a_table_segment.pcd')
    save_pcd(pcl_cloud_objects, '04b_objects_segment.pcd')

    ### Euclidean Clustering
    CLUSTER_TOLERANCE = 0.01
    MIN_CLUSTER_SIZE = 100
    MAX_CLUSTER_SIZE = 4000
    # create copy of objects cloud, convert XYZRGB to XYZ
    # It is dark because it has no color ;)
    dark_cloud = XYZRGB_to_XYZ(pcl_cloud_objects)
    save_pcd(dark_cloud, '05_dark_cloud.pcd')

    # Apply Euclidean clustering and aggregate into single cluster where points
    # are colored by cluster
    cluster_indices_list = cluster_indices(dark_cloud,
                    clusterTolerance=CLUSTER_TOLERANCE,
                    minClusterSize=MIN_CLUSTER_SIZE,
                    maxClusterSize=MAX_CLUSTER_SIZE)
    pcl_clustered_cloud = one_euclidean_cloud_from_indices(dark_cloud, cluster_indices_list)
    save_pcd(dark_cloud, '06_oneEuclideanCloud.pcd')

    ### Create Cluster-Mask Point Cloud to visualize each cluster separately
    pcl_object_clouds = object_clouds_from_indices(pcl_cloud_objects, cluster_indices_list)

    ### Convert PCL data to ROS messages
    ros_objects = [pcl_to_ros(pcl_object_clouds[i]) for i in xrange(0, len(pcl_object_clouds))]

    ros_cloud_objects = pcl_to_ros(pcl_clustered_cloud)
    ros_cloud_table = pcl_to_ros(pcl_cloud_table)

    # Publish ROS messages
    # confusingly, these publishers send ros PointCloud2 data format, NOT pcl
    # the pcl label indicates that it's point cloud data in an abstract sense,
    # and it also indicates that it has been processed with pcl algorithms
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)

    # Exercise-3 TODOs:
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for i, pts_list in enumerate(cluster_indices_list):
        # Grab the points for the cluster
        ros_cloud = ros_objects[i]
        # Compute the associated feature vector
        hsv_hists = compute_color_histograms(ros_cloud, nbins=N_BINS, using_hsv=True)
        rgb_hists = compute_color_histograms(ros_cloud, nbins=N_BINS, using_hsv=False)
        normals = get_normals(ros_cloud)
        nhists = compute_normal_histograms(normals, nbins=N_BINS)
        feature = np.concatenate((rgb_hists, hsv_hists, nhists))
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Place ros_cloud pointer in label dictionary:
        object_ref_dict[label]['ros'] = ros_cloud
        object_ref_dict[label]['pcl'] = pcl_object_clouds[i]
        object_ref_dict[label]['indices'] = pts_list

        # Publish a label into RViz
        label_pos = centroid_of_(label, ros_to_pcl(ros_cloud))
        label_pos[2] += 0.3
        object_markers_pub.publish(make_label(label, label_pos, i))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), sorted(detected_objects_labels)))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    if len(detected_objects_labels) != 0:
        try:
            pr2_mover(detected_objects, pcl_cloud)
        except rospy.ROSInterruptException:
            pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list, pcl_cloud):
    '''
    
    :param object_list: 
    :param pcl_cloud: 
    :return: 
    '''
    global object_list_param
    # TODO: Initialize variables
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    dict_list = []

    # TODO: Parse parameters into individual variables - ?????

    # TODO: Rotate PR2 in place to capture side tables for the collision map
    ABS_ANGLE_TOLERANCE = 0.0017453292519943296 # 0.1 degrees, in radians
    R_TARGET = -np.math.pi/2.0
    L_TARGET = np.math.pi/2.0

    print('Objects: '+str(object_list_param))

    while not r_pivoted:
        joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)
        joint_state_digester(joint_state)
        pivot_bot(R_TARGET, ABS_ANGLE_TOLERANCE, auto_recenter=True)
        time.sleep(1)
    print('r_pivot step complete, value: '+str(r_pivoted))

    while not l_pivoted:
        joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)
        joint_state_digester(joint_state)
        pivot_bot(L_TARGET, ABS_ANGLE_TOLERANCE, auto_recenter=True)
        time.sleep(1)
    print('l_pivot step complete, value: '+str(l_pivoted))

    ### Loop through the pick list
    # for object in detected_objects:
    for i in xrange(0, len(object_list_param)):
        OBJECT_NAME = String()
        OBJECT_NAME.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        for ind, obj in enumerate(object_list):
            if obj.label == OBJECT_NAME.data:
                labels.append(obj.label)
                points_arr = ros_to_pcl(obj.cloud).to_array()
                centroids.append(centroid_of_(OBJECT_NAME.data))

                PICK_POSE = Pose()
                PICK_POSE.position.x = centroids[-1][0]
                PICK_POSE.position.y = centroids[-1][1]
                PICK_POSE.position.z = centroids[-1][2]

                # TODO: Create 'place_pose' for the object
                PLACE_POSE = Pose()
                PLACE_POSE.position.x = -0.02
                PLACE_POSE.position.y = 0.71
                PLACE_POSE.position.z = 0.7
                # PLACE_POSE.orientation.x = 0.0

                # TODO: Assign the arm to be used for pick_place
                ARM_NAME = String()

                if object_group == 'red': # left arm_name
                    ARM_NAME.data = 'left'

                elif object_group == 'green':
                    ARM_NAME.data = 'right'
                    # flip sign of y-coordinate to move to right box
                    PLACE_POSE.position.y *= -1.0

                else:
                    rospy.loginfo("something has gone wrong! group not red or green!")

                ### Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
                yaml_dict = make_yaml_dict(TEST_SCENE_NUM, ARM_NAME, OBJECT_NAME, PICK_POSE, PLACE_POSE)
                dict_list.append(yaml_dict)
                break
        else:
            rospy.loginfo("Object in pick list not found in detected_objects!")

        # Populate various ROS messages
        avoidance_cloud = collision_cloud(OBJECT_NAME.data, pcl_cloud)
        collision_point_pub.publish(avoidance_cloud)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, ARM_NAME, PICK_POSE, PLACE_POSE)
            print("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml("output_{}.yaml".format(TEST_SCENE_NUM), dict_list)





if __name__ == '__main__':
    if save_clouds_to_pcd:
        pcd_list = glob.glob(save_dir+'*.pcd')

    ### Load Model From disk - done first to save ROS overhead in case of failure
    # model = {
    #  'classes': <list of class name strings>,
    #  'classifier': <calls sklearn.svm.SVC() class instantiation>
    #  'scaler': <calls sklearn.preprocessing.StandardScaler() class instantiation>
    # } # classifier and scaler generated w/saved presets

    if len(sys.argv) > 1:
        scene_number = int(sys.argv[1])

    # TODO: FINISH REMOVING STUPID BREADCRUMB LOGIC
    elif len(glob.glob('*.breadcrumb')) > 0:
        scene_number = int(glob.glob('*.breadcrumb')[0][0])
        
    else:
        raise RuntimeError("Could not determine scene to load!")
    
    assert isinstance(scene_number, int)
    assert scene_number in (1, 2, 3)

    TEST_SCENE_NUM = Int32()
    TEST_SCENE_NUM.data = scene_number
    
    print("\nTest scene number: "+str(TEST_SCENE_NUM))
    
    model_glob_pattern = 'fullmodel_*_o*_h*.sav'
    model_file_match = glob.glob(save_dir+model_glob_pattern)
    
    QUIT_OUT = False # a failthrough option in case there is an issue loading model
    
    if not model_file_match:
        print('No model matching glob pattern "{}" was detected!'.format(model_glob_pattern))
        while True:
#            model_filename = str(input("enter target model filename manually, or press enter to quit: "))
            model_filename = raw_input("enter target model filename manually, or press enter to quit: ")
            print model_filename
            if model_filename.strip() == "":
                print('exiting project_template.py... ')
                QUIT_OUT = True
                break

            elif model_filename[-4:] == '.sav':
                model_file_match = glob.glob(model_filename)
                if model_file_match:
                    break
                else:
                    print('no results were found, try again!')
                    continue
            else:
                print('invalid filename, try again!')
                continue
    else:
        pass

    if not QUIT_OUT:
        while len(model_file_match) > 1:
            print("\nFound multiple models, expected one!")
            print("Options: ")
            for ind, model_file in enumerate(model_file_match):
                print(str(ind)+' - '+model_file)
            input_index = int(eval(str(input("select a model by entering its index here: "))))
            if 0 <= input_index < len(model_file_match):
                model_file_match = [model_file_match[input_index]]
            else:
                print("invalid input; out of index bounds!")

        model = pickle.load(open(model_file_match[0], 'rb'))
        N_BINS = int(eval(model_file_match[0].split('_')[3][1:-4]))

        clf = model['classifier']
        encoder = LabelEncoder()
        encoder.classes_ = model['classes']
        scaler = model['scaler']

        print("\nFound and loaded model from file: "+model_file_match[0])

        ### ROS node initialization
        rospy.init_node('clustering', anonymous=True)

        r_pivoted = False
        l_pivoted = False

        world_joint_pos = 0.0
        world_joint_moving = False

        ### Get/Read parameters
        object_list_param = rospy.get_param('/object_list')

        ### create object centroid tracker
        tracked_centroids = {}

        ### tracker for items successfully pick-place'd
        completed_transports = []

        ###
        object_ref_dict = {
            'sticky_notes': {'ros': None, 'pcl': None, 'indices': None},
            'book':         {'ros': None, 'pcl': None, 'indices': None},
            'snacks':       {'ros': None, 'pcl': None, 'indices': None},
            'biscuits':     {'ros': None, 'pcl': None, 'indices': None},
            'eraser':       {'ros': None, 'pcl': None, 'indices': None},
            'soap2':        {'ros': None, 'pcl': None, 'indices': None},
            'soap':         {'ros': None, 'pcl': None, 'indices': None},
            'glue':         {'ros': None, 'pcl': None, 'indices': None},
        }


        ### Create Subscribers
        # pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
        pcl_sub = rospy.Subscriber("/pr2/world/points", PointCloud2, pcl_callback, queue_size=1)
        # joint_state_sub = rospy.Subscriber('/pr2/joint_states', JointState, jointstate_callback, queue_size=1)

        ### Create Publishers - What am I supposed to be publishing here? Do I need more still?
        # Table and Objects Clouds
        pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
        pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
        # Detected Object List and Marker List
        detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
        object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
        # World Joint (rotates base link of pr2)
        world_joint_pub = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=1)
        # Collision points for the pick-place routine planning algorithm
        collision_point_pub = rospy.Publisher('/pr2/3d_map/points', PointCloud2, queue_size=1)

        ### Initialize color_list
        get_color_list.color_list = []

        ### Spin while node is not shutdown
        while not rospy.is_shutdown():
            rospy.spin()

    else:
        pass
