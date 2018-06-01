import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from pcl_helper import *


def rgb_to_hsv(rgb_list):
    '''

    :param rgb_list:
    :return:
    '''
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, nbins=64, using_hsv=False):
    '''

    :param cloud:
    :param nbins:
    :param using_hsv:
    :return:
    '''

    # ''':param cloud: PointCloud2 data from RGBD detection of object
    # :param using_hsv: whether or not to convert the RGB color values to HSV values in pre-processing
    # :return: normalized feature vector'''

    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    point_colors_array = np.zeros((3, len(point_colors_list)))

    for i, color in enumerate(point_colors_list):
        point_colors_array[:, i] = color[:] # channels fed through in parallel

    # for i, color in enumerate(point_colors_list):
    #     point_colors_array[0][i] = color[0] # channel 1
    #     point_colors_array[1][i] = color[1] # channel 2
    #     point_colors_array[2][i] = color[2] # channel 3
    
    # TODO: Compute histograms
    channel_1_hist = np.histogram(point_colors_array[0, :], bins=nbins, range=(0, 256))
    channel_2_hist = np.histogram(point_colors_array[1, :], bins=nbins, range=(0, 256))
    channel_3_hist = np.histogram(point_colors_array[2, :], bins=nbins, range=(0, 256))

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_3_hist[0])).astype(np.float64)

    normed_features = hist_features / np.sum(hist_features)

    return normed_features 



def compute_normal_histograms(normal_cloud, nbins=64):
    '''

    :param normal_cloud:
    :param nbins:
    :return:
    '''
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):

        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])


    # TODO: Compute histograms of normal values (just like with color)
    norm_x_hist = np.histogram(norm_x_vals, bins=nbins, range=(0, 256))
    norm_y_hist = np.histogram(norm_y_vals, bins=nbins, range=(0, 256))
    norm_z_hist = np.histogram(norm_z_vals, bins=nbins, range=(0, 256))

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features)

    return normed_features
