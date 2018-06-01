#!/usr/bin/env python

# Import modules
import os
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


MODEL_GLOB_PATTERN = 'fullmodel_*_o*_h*.sav'
SAVE_DIR = './savefiles/'
SAVE_CLOUDS = True
DO_AUTO_PIVOT = False

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

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
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

#
def send_to_yaml(yaml_file_path, dict_list):
    '''
    Helper function to output data to yaml file

    :param yaml_file_path: where to save to; output_n.yaml
    :param dict_list:
    :return:
    '''
    data_dict = {"object_list": dict_list}
    with open(yaml_file_path, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def load_svm_model():
    '''
    Loads pre-fit support vector machine from file.

    model = {
     'classes': list of class name strings>
     'classifier': calls sklearn.svm.SVC() class instantiation
     'scaler': calls sklearn.preprocessing.StandardScaler() class instantiation
    } # classifier and scaler generated w/saved presets

    :return: model, n_bins if successful, else None
    '''
    model_file_match = glob.glob(SAVE_DIR + MODEL_GLOB_PATTERN)

    quit_out = False  # a failthrough option in case there is an issue loading model

    if not model_file_match:
        print('No model matching glob pattern "{}" was detected!'.format(MODEL_GLOB_PATTERN))
        while True:
            #            model_filename = str(input("enter target model filename manually, or press enter to quit: "))
            model_filename = raw_input("enter target model filename manually, or press enter to quit: ")
            # print model_filename
            if model_filename.strip() == "":
                print('exiting project_template.py... ')
                quit_out = True
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

    if not quit_out:
        while len(model_file_match) > 1:
            print("\nFound multiple models, expected one!")
            print("Options: ")
            for ind, model_file in enumerate(model_file_match):
                print(str(ind) + ' - ' + model_file)
            input_index = int(eval(str(input("select a model by entering its index here: "))))
            if 0 <= input_index < len(model_file_match):
                model_file_match = [model_file_match[input_index]]
            else:
                print("invalid input; out of index bounds!")

        model = pickle.load(open(model_file_match[0], 'rb'))
        n_bins = int(eval(model_file_match[0].split('_')[3][1:-4]))

        rospy.loginfo("Found and loaded model from file: " + model_file_match[0])
        return model, n_bins
    return None



class PCDSaver:
    def __init__(self, output_directory, is_active):
        assert isinstance(output_directory, str)
        assert os.path.isdir(output_directory)
        assert isinstance(is_active, bool)

        self.is_active = is_active
        self.output_directory = output_directory
        self.pcd_list_on_init = glob.glob(output_directory + '*.pcd')
        self.pcd_list = self.pcd_list_on_init[:]
        self.pcd_files_written = []
        self.requests_not_fulfilled = []

        self.locked = True

    def dump(self, pcl_cloud, filename):
        if not self.is_active:
            pass
        elif self.locked:
            pass
        elif filename not in self.pcd_list:
            pcl.save(pcl_cloud, self.output_directory + filename)
            self.pcd_list.append(filename)
            self.pcd_files_written.append(filename)
            rospy.loginfo('Saved point cloud file: {}'.format(filename))
        else:
            self.requests_not_fulfilled.append(filename)



class CloudPipeline:
    save_dir = SAVE_DIR
    save_clouds_to_pcd = SAVE_CLOUDS
    auto_pivot_enabled = DO_AUTO_PIVOT

    def __init__(self, object_list_param, model, n_bins, test_scene_num):
        self.object_list_param = object_list_param
        self.n_objects = len(object_list_param)
        self.scene_number = test_scene_num
        # Set up SVM model
        self.model = model
        self.n_hist_bins = n_bins
        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']
        # Initialize non-parameter values/objects
        self.gazed_cycles_completed = 0
        # PR2 control state
        self.world_joint_pos = 0.0
        self.world_joint_moving = False
        self.l_pivoted = False
        self.r_pivoted = False
        # yaml saving redundancy removal
        self.yaml_saved = False
        # Set up PCD saver
        self.saver = PCDSaver(CloudPipeline.save_dir, CloudPipeline.save_clouds_to_pcd)
        self.save_pcd = self.saver.dump

        # Set up various centralized data objects
        self.tracked_centroids = {}
        # self.completed_transports = []
        self.object_ref_dict = {
            'sticky_notes': {'ros': None, 'pcl': None, 'indices': None},
            'book':         {'ros': None, 'pcl': None, 'indices': None},
            'snacks':       {'ros': None, 'pcl': None, 'indices': None},
            'biscuits':     {'ros': None, 'pcl': None, 'indices': None},
            'eraser':       {'ros': None, 'pcl': None, 'indices': None},
            'soap2':        {'ros': None, 'pcl': None, 'indices': None},
            'soap':         {'ros': None, 'pcl': None, 'indices': None},
            'glue':         {'ros': None, 'pcl': None, 'indices': None},
        }

        ################################################
        # PIPELINE PARAMS
        # passthrough filter bounds
        self.z_ax_min = 0.4
        self.z_ax_max = 0.9
        self.y_ax_min = -0.4
        self.y_ax_max = 0.4
        self.x_ax_min = 0.35
        self.x_ax_max = 0.9

        # Statistical Outlier Filtering
        # First pass
        self.mean_k_1 = 10
        self.threshold_scale_1 = 0.8
        # second pass
        self.mean_k_2 = 10
        self.threshold_scale_2 = 1.6

        ### Voxel Grid Downsampling
        # Choose a voxel (also known as leaf) size
        self.leaf_scale = 0.005  # meters per voxel_edge - this will affect the required level for MIN_CLUSTER_SIZE
        self.xyz_voxel_coeffs = (1.0, 1.0, 1.0)  # scale xyz dimensions of voxel independently

        ### RANSAC Plane Segmentation separation tolerance
        self.max_dist = 0.0025

        ### Euclidean Clustering
        # THESE VALUES MUST BE TUNED! They will have different levels of viability depending somewhat on aggressiveness
        # of stat outlier filtering, and very strongly on the scale of voxel downsampling
        self.cluster_tolerance = 0.01
        self.min_cluster_size = 100
        self.max_cluster_size = 4000

        # Parameters to control
        self.gaze_frames = 20

        self.abs_angle_tolerance = 0.0017453292519943296  # 0.1 degrees, in radians
        self.r_pivot_target = -np.math.pi / 2.0
        self.l_pivot_target = np.math.pi / 2.0

        rospy.loginfo("Perception Pipeline initialized at {}".format(time.strftime('%c')))
        rospy.loginfo("Test scene number: " + str(self.scene_number.data))
        for i in xrange(self.n_objects):
            rospy.loginfo('pick list item {}: {}'.format(i, self.object_list_param[i]))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def passthrough_filter(pcl_cloud, axis_min=0.7, axis_max=1.2, filter_axis='z'):
        '''

        :param pcl_cloud:
        :param axis_min:
        :param axis_max:
        :param filter_axis:
        :return:
        '''
        passthrough = pcl_cloud.make_passthrough_filter()
        # Assign axis and range to the passthrough filter object.
        passthrough.set_filter_field_name(filter_axis)
        # Recommended/Example settings: min: 0.6, max: 1.1
        passthrough.set_filter_limits(axis_min, axis_max)
        # Finally use the filter function to obtain the resultant point cloud.
        cloud_filtered = passthrough.filter()
        return cloud_filtered

    @staticmethod
    def ransac_extract(pcl_cloud, max_distance=0.01):
        '''

        :param pcl_cloud:
        :param max_distance:
        :return:
        '''
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

    @staticmethod
    def cluster_indices(dark_cloud, cluster_tolerance=0.01, min_cluster_size=40, max_cluster_size=500):
        '''

        :param dark_cloud:
        :param cluster_tolerance:
        :param min_cluster_size:
        :param max_cluster_size:
        :return:
        '''
        # construct a k-d tree from the cloud_objects point cloud
        tree = dark_cloud.make_kdtree()
        # Create a cluster extraction object
        ec = dark_cloud.make_EuclideanClusterExtraction()
        # Set tolerances for distance threshold
        # as well as minimum and maximum cluster size (in points)
        ec.set_ClusterTolerance(cluster_tolerance)
        ec.set_MinClusterSize(min_cluster_size)
        ec.set_MaxClusterSize(max_cluster_size)
        # Search the k-d tree for clusters
        ec.set_SearchMethod(tree)
        # Extract indices for each of the discovered clusters
        obj_cluster_indices = ec.Extract()
        return obj_cluster_indices

    @staticmethod
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

    @staticmethod
    def object_clouds_from_indices(pcl_cloud, obj_cluster_indices_list, negative_toggle=False):
        '''

        :param pcl_cloud:
        :param obj_cluster_indices_list:
        :param negative_toggle:
        :return:
        '''
        # Assign each point a color based on its cluster affiliation and append to
        # unified list of colored points

        # cluster_list = []

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


    def centroid_of_(self, object_name, pcl_cloud=None, cast_python_float=True):
        '''

        :param object_name:
        :param pcl_cloud:
        :param cast_python_float:
        :return:
        '''
        if pcl_cloud:
            self.tracked_centroids[object_name] = np.mean(pcl_cloud, axis=0)[:3]

        if cast_python_float:
            return [np.asscalar(self.tracked_centroids[object_name][i]) for i in (0, 1, 2)]

        return self.tracked_centroids[object_name]


    def collision_cloud(self, target_item_name, pcl_cloud):
        '''
        :param target_item_name:
        :param pcl_cloud:
        :return:
        '''
        # pcl_cloud here should be from before the ransac segmentation step
        collidable = pcl_cloud
        # for all objects loaded into scene
        for item in self.object_list_param:
            # if an item has already been pick-placed or if it is our target...
            if item['name'] == target_item_name:
                # then exclude it from collision model by extracting its indices
                collidable = collidable.extract(self.object_ref_dict[item['name']]['indices'], negative=True)
        self.save_pcd(collidable, '07_collision_cloud.pcd')
        ros_collision_cloud = pcl_to_ros(collidable)
        return ros_collision_cloud


    def joint_state_digester(self, joint_state_data, abs_ang_delta_tolerance=0.01):
        '''

        :param joint_state_data:
        :param abs_ang_delta_tolerance:
        :return:
        '''
        angle_delta = joint_state_data.position[-1] - self.world_joint_pos
        self.world_joint_pos = joint_state_data.position[-1]
        self.world_joint_moving = (np.absolute(angle_delta) > abs_ang_delta_tolerance)

    # bot pivot manager
    def pivot_bot(self, world_joint_goal, absolute_angle_tolerance, auto_recenter=True):
        '''

        :param world_joint_goal:
        :param absolute_angle_tolerance:
        :param auto_recenter:
        :return:
        '''
        sgn = np.sign(world_joint_goal)
        output_text = ''
        task_complete = False

        if self.world_joint_moving:
            # return 'joint in motion...'
            return
        else:
            pass

        if sgn == -1:  # If pivoting right:
            if self.r_pivoted:
                task_complete = True
            elif not self.world_joint_pos <= ((world_joint_goal) + absolute_angle_tolerance):
                world_joint_pub.publish(world_joint_goal)
                output_text += 'pivoting right to {}; '.format(round(world_joint_goal, 5))
            else:
                self.r_pivoted = True
                task_complete = True
                output_text += 'pivot right complete! '
        elif sgn == 1:  # If pivoting left:
            if self.l_pivoted:
                task_complete = True
            elif not self.world_joint_pos >= ((world_joint_goal) - absolute_angle_tolerance):
                world_joint_pub.publish(world_joint_goal)
                output_text += 'pivoting left to {}; '.format(round(world_joint_goal, 5))
            else:
                self.l_pivoted = True
                task_complete = True
                output_text += 'pivot left complete! '
        else:
            pass
        if (auto_recenter and task_complete) or sgn == 0:
            if np.fabs(self.world_joint_pos) > absolute_angle_tolerance:
                world_joint_pub.publish(0.0)
                if auto_recenter:
                    output_text += 'automatically re-centering robot.'
                else:
                    output_text += 'pivoting to centered position...'

        elif np.fabs(self.world_joint_pos) < absolute_angle_tolerance:
            output_text += 'beginning pivot!'
        else:
            output_text += 'halted at target orientation away from center.'

        rospy.loginfo(output_text)


    # Callback function for your Point Cloud Subscriber
    def pcl_callback(self, pcl_msg):
        '''
        :param pcl_msg:
        :return:
        '''
        ### Convert ROS msg to PCL data
        pcl_cloud = ros_to_pcl(pcl_msg)

        # save_pcd is a conditionally-active file output utility
        self.save_pcd(pcl_cloud, '00_raw_cloud.pcd')

        ### PassThrough Filter - chops the data down by a huge factor before proceeding
        # passthrough filter out the approximate height of the table surface and objects
        pcl_cloud = self.passthrough_filter(pcl_cloud, axis_min=self.z_ax_min, axis_max=self.z_ax_max, filter_axis='z')
        # passthrough filter out the xy-range which corresponds to the table
        pcl_cloud = self.passthrough_filter(pcl_cloud, axis_min=self.y_ax_min, axis_max=self.y_ax_max, filter_axis='y')
        pcl_cloud = self.passthrough_filter(pcl_cloud, axis_min=self.x_ax_min, axis_max=self.x_ax_max, filter_axis='x')

        self.save_pcd(pcl_cloud, '01_passthrough_filtered.pcd')

        ### Statistical Outlier Filtering
        pcl_cloud = self.stat_outlier_filter(pcl_cloud, mean_k=self.mean_k_1, threshold_scale=self.threshold_scale_1)
        self.save_pcd(pcl_cloud, '02a_outlier_filtered.pcd')

        pcl_cloud = self.stat_outlier_filter(pcl_cloud, mean_k=self.mean_k_2, threshold_scale=self.threshold_scale_2)
        self.save_pcd(pcl_cloud, '02b_outlier_refiltered.pcd')

        ### Voxel Grid Downsampling
        # Choose a voxel (also known as leaf) size
        # LEAF_SCALE = 0.005  # meters per voxel_edge - this will affect the required level for MIN_CLUSTER_SIZE
        # XYZ_VOXEL_COEFFS = (1.0, 1.0, 1.0)  # scale xyz dimensions of voxel independently
        pcl_cloud = self.voxel_downsample(pcl_cloud, leaf_scale=self.leaf_scale, coeffs=self.xyz_voxel_coeffs)
        self.save_pcd(pcl_cloud, '03_voxel_downsampled.pcd')

        ### RANSAC Plane Segmentation
        pcl_cloud_table, pcl_cloud_objects = self.ransac_extract(pcl_cloud, max_distance=self.max_dist)

        self.save_pcd(pcl_cloud_table, '04a_table_segment.pcd')
        self.save_pcd(pcl_cloud_objects, '04b_objects_segment.pcd')

        ### Euclidean Clustering
        # create copy of objects cloud, convert XYZRGB to XYZ
        # It is dark because it has no color ;)
        dark_cloud = XYZRGB_to_XYZ(pcl_cloud_objects)
        self.save_pcd(dark_cloud, '05_dark_cloud.pcd')

        # Apply Euclidean clustering and aggregate into single cluster where points
        # are colored by cluster
        cluster_indices_list = self.cluster_indices(dark_cloud,
                                                    cluster_tolerance=self.cluster_tolerance,
                                                    min_cluster_size=self.min_cluster_size,
                                                    max_cluster_size=self.max_cluster_size)

        pcl_clustered_cloud = self.one_euclidean_cloud_from_indices(dark_cloud, cluster_indices_list)
        self.save_pcd(dark_cloud, '06_oneEuclideanCloud.pcd')

        ### Create Cluster-Mask Point Cloud to visualize each cluster separately
        pcl_object_clouds = self.object_clouds_from_indices(pcl_cloud_objects, cluster_indices_list)

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

        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels = []
        detected_objects = []
        for i, pts_list in enumerate(cluster_indices_list):
            # Grab the points for the cluster
            ros_cloud = ros_objects[i]
            # Compute the associated feature vector
            hsv_hists = compute_color_histograms(ros_cloud, nbins=self.n_hist_bins, using_hsv=True)
            rgb_hists = compute_color_histograms(ros_cloud, nbins=self.n_hist_bins, using_hsv=False)
            normals = get_normals(ros_cloud)
            nhists = compute_normal_histograms(normals, nbins=self.n_hist_bins)
            feature = np.concatenate((rgb_hists, hsv_hists, nhists))
            # Make the prediction
            prediction = self.clf.predict(self.scaler.transform(feature.reshape(1, -1)))
            label = self.encoder.inverse_transform(prediction)[0]
            detected_objects_labels.append(label)

            # Place ros_cloud pointer in label dictionary:
            self.object_ref_dict[label]['ros'] = ros_cloud
            self.object_ref_dict[label]['pcl'] = pcl_object_clouds[i]
            self.object_ref_dict[label]['indices'] = pts_list

            # Publish a label into RViz
            label_pos = self.centroid_of_(label, ros_to_pcl(ros_cloud))
            label_pos[2] += 0.2

            object_markers_pub.publish(make_label(label, label_pos, i))
            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cloud
            detected_objects.append(do)

        # If the pcd saver is still locked from initialization/warm-up, unlock it when all objects are detected
        if self.saver.locked and len(detected_objects_labels) == self.n_objects:
            self.saver.locked = False

        rospy.loginfo('Detected {}/{} objects: {}'.format(len(detected_objects_labels),
                                                          self.n_objects,
                                                          sorted(detected_objects_labels)))

        # Publish the list of detected objects
        detected_objects_pub.publish(detected_objects)
        # Could add some logic to determine whether or not object detections are robust before calling pr2_mover()
        if len(detected_objects_labels) != 0:
            try:
                self.pr2_mover(detected_objects, pcl_cloud)
            except rospy.ROSInterruptException:
                pass


    # function to load parameters and request PickPlace service
    def pr2_mover(self, detected_object_list, pcl_cloud):
        '''
        :param detected_object_list: detected_object_list from svm model classifier
        :param pcl_cloud: point cloud containing all objects and table surface, for producing collision avoidance clouds
        :return:
        '''
        # Before rotating PR2 in place, stare at table to check basic pipeline functionality & generate data/outputs
        if self.gazed_cycles_completed < self.gaze_frames:
            # time.sleep(0.25)
            self.gazed_cycles_completed += 1
        # If rotation is turned off or completed, skip ahead!
        elif (not CloudPipeline.auto_pivot_enabled) or (self.l_pivoted and self.r_pivoted):
            pass
        # If right rotation is not completed, advance the rightward rotation process
        elif not self.r_pivoted:
            joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)
            self.joint_state_digester(joint_state)
            self.pivot_bot(self.r_pivot_target, self.abs_angle_tolerance, auto_recenter=True)
        # If left rotation is not completed, advance the leftward rotation process
        elif not self.l_pivoted:
            joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)
            self.joint_state_digester(joint_state)
            self.pivot_bot(self.l_pivot_target, self.abs_angle_tolerance, auto_recenter=True)

        # If rotation is in progress, skip subsequent data processing
        if not self.world_joint_moving:
            # Initialize variables
            labels = []
            centroids = []  # to be list of tuples (x, y, z)
            dict_list = []
            missing = []
            for i in xrange(0, len(self.object_list_param)):
                object_name = String()
                object_name.data = self.object_list_param[i]['name']
                object_group = self.object_list_param[i]['group']
                # Get the PointCloud for a given object and obtain it's centroid
                # since the object list isn't ordered the same as the pick list, we do a quick loop over the list of
                # detected objects looking for a matching label:
                for ind, obj in enumerate(detected_object_list):
                    # if a given object in detected_objects matches our pick list object, enter into next step...
                    if obj.label == object_name.data:
                        # first, append the object label to our labels list
                        labels.append(obj.label)
                        # calculate centroid of cluster point cloud corresponding to that label;
                        # see method: CloudPipeline.centroid_of_ for detail
                        centroids.append(self.centroid_of_(object_name.data))
                        # create 'pick_pose' for the object
                        pick_pose = Pose()
                        pick_pose.position.x = centroids[-1][0]
                        pick_pose.position.y = centroids[-1][1]
                        pick_pose.position.z = centroids[-1][2]
                        # Create 'place_pose' for the object - these values were determined manually and may not be ideal.
                        place_pose = Pose()
                        place_pose.position.x = -0.02
                        place_pose.position.y = 0.71
                        place_pose.position.z = 0.7

                        # Create arm name object and assign the arm to be used for pick_place
                        arm_name = String()
                        if object_group == 'red':  # left arm_name
                            arm_name.data = 'left'
                        else:
                            arm_name.data = 'right'
                            # flip sign of y-coordinate to move to right box
                            place_pose.position.y *= -1.0

                        # Make a list of dictionaries for later output to yaml format if all objects are found
                        if not self.yaml_saved and len(detected_object_list) == self.n_objects:
                            yaml_dict = make_yaml_dict(self.scene_number, arm_name, object_name, pick_pose, place_pose)
                            dict_list.append(yaml_dict)

                        # Output calculated request parameters into output yaml file
                        break # stop looping, we already processed the object we were looking for

                else: # if object not found, thus no break triggered, add to missing list
                    missing.append(object_name.data)

                # Generate and publish ROS message for collision avoidance
                avoidance_cloud = self.collision_cloud(object_name.data, pcl_cloud)
                collision_point_pub.publish(avoidance_cloud)

            if not self.yaml_saved and not missing:
                send_to_yaml(self.save_dir + "output_{}.yaml".format(self.scene_number.data), dict_list)
                rospy.loginfo("Saved information to yaml file!".format(missing))
                self.yaml_saved = True

            # # Wait for 'pick_place_routine' service to come up
            # rospy.wait_for_service('pick_place_routine')
            # try:
            #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            #     # TODO: Insert your message variables to be sent as a service request
            #     resp = pick_place_routine(self.scene_number, object_name, arm_name, pick_pose, place_pose)
            #     print("Response: ", resp.success)
            # except rospy.ServiceException, e:
            #     print "Service call failed: %s" % e


if __name__ == '__main__':
    print("\n")

    TEST_SCENE_NUM = Int32()

    if len(sys.argv) > 1:
        TEST_SCENE_NUM.data = int(sys.argv[1])
        assert isinstance(TEST_SCENE_NUM.data, int)
        assert TEST_SCENE_NUM.data in (1, 2, 3)

    else:
        raise RuntimeError("Could not determine scene to load!")

    # Load Model From disk - done first to save ROS overhead in case of failure
    model_load_result = load_svm_model()

    if model_load_result is not None:
        svm_model, n_bins = model_load_result

        ### ROS node initialization
        rospy.init_node('clustering', anonymous=True)

        ### Get/Read parameters
        object_list_param = rospy.get_param('/object_list')

        pipeline = CloudPipeline(object_list_param, svm_model, n_bins, TEST_SCENE_NUM)

        ### Create Subscribers
        # pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
        pcl_sub = rospy.Subscriber("/pr2/world/points", PointCloud2, pipeline.pcl_callback, queue_size=1)
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
