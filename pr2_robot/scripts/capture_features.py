#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
# from sensor_stick.training_helper import spawn_model
# from sensor_stick.training_helper import delete_model
# from sensor_stick.training_helper import initial_setup
# from sensor_stick.training_helper import capture_sample
from training_helper import spawn_model
from training_helper import delete_model
from training_helper import initial_setup
from training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


histogram_bins_record = True
listnumber = 1


pick_list_ = {1: ['biscuits', 'soap', 'soap2'],
              2: ['biscuits', 'soap', 'soap2', 'glue'],
              3: ['sticky_notes', 'book', 'snacks', 'biscuits',
                  'eraser', 'soap2', 'soap', 'glue']
             }

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':

    n_orientations = 30
    histogram_bins = 64

    rospy.init_node('capture_node')

    while True:
        n_orients_input = str(input("enter number of orientations per object or enter for default ({}): ".format(n_orientations)))
        if n_orients_input.strip() == "":
            print('using default: '+str(n_orientations))
            break

        try:
            n_orientations = int(eval(n_orients_input.strip()))
        except:
            print("couldn't convert input to integer; try again!")
            continue

        if n_orientations <= 0:
            print('invalid input - enter a nonzero, positive integer!')
            continue
        elif n_orientations > 20:
            print("capture process may take some time; go stretch your legs!")
            break
        else:
            print(str(n_orientations)+" orientations is probably too few to properly train SVM, but here we go!")
            break


    while True:
        hist_bins_input = str(input("enter number feature histogram bins, or enter for default ({}): ".format(histogram_bins)))
        if hist_bins_input.strip() == "":
            print('using default: '+str(histogram_bins))
            break

        try:
            histogram_bins = int(eval(hist_bins_input.strip()))
        except:
            print("couldn't convert input to integer; try again!")
            continue

        if histogram_bins <= 0:
            print('invalid input - enter a nonzero, positive integer!')
            continue
        elif histogram_bins > 256:
            print('chose a number of bins: 0 < n_bins <= 256')
            continue
        else:
            break

    if histogram_bins_record:
        with open('model_hist_bin_count.txt', 'w') as record:
            record.writeline(str(histogram_bins))
    else:
        pass

    object_models = pick_list_[listnumber]

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in object_models:
        spawn_model(model_name)
        print(model_name)
        for i in range(n_orientations):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            hsv_hists = compute_color_histograms(sample_cloud, nbins=histogram_bins, using_hsv=True)
            rgb_hists = compute_color_histograms(sample_cloud, nbins=histogram_bins, using_hsv=False)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals, nbins=histogram_bins)
            feature = np.concatenate((rgb_hists, hsv_hists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()


    pickle.dump(labeled_features, open('training_set.sav', 'wb'))
