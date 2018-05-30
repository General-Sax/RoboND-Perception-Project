import pcl
import rospy

from sensor_stick.srv import GetNormals
from sensor_stick.pcl_helper import *
from sensor_stick.marker_tools import *




# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

def statOutlierFilter(pcl_cloud, mean_k=100, threshold_scale=0.5):
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

def objClouds_fromIndices(pcl_cloud, obj_cluster_indices_list, negative_toggle=False):
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

def pcdRecord(pcl_cloud, filename):
    global pcd_list
    if filename not in pcd_list:
        pcd_list.append(filename)
        pcl.save(pcl_cloud, save_dir+filename)
    else:
        pass

def centroid_of_(object_name, pcl_cloud=None, cast_python_float=True):
    global trackedCentroids
    if pcl_cloud:
        trackedCentroids[object_name] = np.mean(pcl_cloud, axis=0)[:3]

    if cast_python_float:
        return [np.asscalar(trackedCentroids[object_name][i]) for i in (0, 1, 2)]

    return trackedCentroids[object_name]

def collision_cloud(target_item_name, pcl_cloud):
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
    if saveClouds: pcdRecord(collidable, '07_collision_cloud.pcd')
    ros_collision_cloud = pcl_to_ros(collidable)
    return ros_collision_cloud

def joint_state_digester(joint_state_data, abs_ang_delta_tolerance=0.01):
    global world_joint_pos, world_joint_moving

    angle_delta = joint_state_data.position[-1] - world_joint_pos
    world_joint_pos = joint_state_data.position[-1]
    world_joint_moving = (np.absolute(angle_delta) > abs_ang_delta_tolerance)

    print("\nworld joint delta: "+str(angle_delta))
    print("considered moving: "+str(world_joint_moving))

# bot pivot manager
def pivot_bot(world_joint_goal, absolute_angle_tolerance, auto_recenter=True):
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

