#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def statOutlierFilter(pcl_cloud, mean_k=50, threshold_scale=1.0):
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
    # leaf_scale: sets default scale for voxel edge, units of meters
    # coeffs: linear scaling factors for edges x, y, z respectively
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

def oneEuclideanCloud_fromIndices(dark_cloud, obj_cluster_indices_list):
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

def objClouds_fromIndices(pcl_cloud, obj_cluster_indices_list):
    # Assign each point a color based on its cluster affiliation and append to
    # unified list of colored points
    cluster_list = []
    pcl_cloud_list = []
    for j, indices in enumerate(obj_cluster_indices_list):
        pcl_cloud_list.append(pcl_cloud.extract(indices))
        # for indice in indices:
        #     cluster_list[-1].append([pcl_cloud[indice][0],
        #                             pcl_cloud[indice][1],
        #                             pcl_cloud[indice][2],
        #                             pcl_cloud[indice][3]])
        # pcl_cloud_gen = pcl.PointCloud_PointXYZRGB()
        # pcl_cloud_gen.from_list(cluster_list[-1])
        # pcl_cloud_list.append(pcl_cloud_gen)
    return pcl_cloud_list

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    ### Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering

    # TODO: Voxel Grid Downsampling

    # TODO: PassThrough Filter

    # TODO: RANSAC Plane Segmentation

    # TODO: Extract inliers and outliers

    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages

    # TODO: Publish ROS messages
    ### Statistical Outlier Filtering
    MEAN_K = 80
    THRESHOLD_SCALE = 0.8
    pcl_cloud = statOutlierFilter(pcl_cloud, mean_k=MEAN_K, threshold_scale=THRESHOLD_SCALE)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)

        # Grab the points for the cluster

        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':
    ### ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    ### Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers - What am I supposed to be publishing here?
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)

    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)


    ### Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    ### Initialize color_list
    get_color_list.color_list = []

    ### Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
