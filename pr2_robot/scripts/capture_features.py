#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
from training_helper import spawn_model
from training_helper import delete_model
from training_helper import initial_setup
from training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

# Where to put the training_set pickle when it's been cooked up
SAVE_DIR = './savefiles/'

# Toggles use of the cli setup logic to customize parameters
UI_PROMPTING = True

# Single complete list of all objects in the project

PROJECT_MODELS = [
    'sticky_notes',
    'book',
    'snacks',
    'biscuits',
    'eraser',
    'soap2',
    'soap',
    'glue',
]

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    # set defaults
    ORIENTATIONS_PER_OBJECT = 30
    N_HIST_BINS = 64

    rospy.init_node('capture_node')

    while UI_PROMPTING:
        n_orients_input = str(input(
            "enter number of orientations per object or <enter> for default ({}): ".format(ORIENTATIONS_PER_OBJECT)))
        
        if n_orients_input.strip() == "":
            print('using default: ' + str(ORIENTATIONS_PER_OBJECT))
            break
            
        try:
            ORIENTATIONS_PER_OBJECT = int(eval(n_orients_input.strip()))
        except:
            print("couldn't convert input to integer; try again!")
            continue

        if ORIENTATIONS_PER_OBJECT <= 0:
            print('invalid input - enter a nonzero, positive integer!')
            continue
        elif ORIENTATIONS_PER_OBJECT > 20:
            print("capture process may take some time; go stretch your legs!")
            break
        else:
            print(str(ORIENTATIONS_PER_OBJECT) + " orientations is probably too few to properly train SVM, but here we go!")
            break

    while UI_PROMPTING:
        hist_bins_input = str(input("enter number feature histogram bins, or enter for default ({}): ".format(N_HIST_BINS)))
        if hist_bins_input.strip() == "":
            print('using default: ' + str(N_HIST_BINS))
            break

        try:
            N_HIST_BINS = int(eval(hist_bins_input.strip()))
        except:
            print("couldn't convert input to integer; try again!")
            continue

        if N_HIST_BINS <= 0:
            print('invalid input - enter a nonzero, positive integer!')
            continue
        elif N_HIST_BINS > 256:
            print('chose a number of bins: 0 < n_bins <= 256')
            continue
        else:
            break

    # Disable gravity and delete the ground plane
    initial_setup()

    labeled_features = []

    for ind, model_name in enumerate(models):
        delete_model()
        spawn_model(model_name)

        for i in range(ORIENTATIONS_PER_OBJECT):
            pct = ((ind * ORIENTATIONS_PER_OBJECT) + i) / (8 * ORIENTATIONS_PER_OBJECT)
            print(str(pct) +"%% - [ " + model_name +" : model " + str(ind+1) +" of 8 ] orientation ( " + str(i) +" of " + str(ORIENTATIONS_PER_OBJECT) + " )")
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
            hsv_hists = compute_color_histograms(sample_cloud, nbins=N_HIST_BINS, using_hsv=True)
            rgb_hists = compute_color_histograms(sample_cloud, nbins=N_HIST_BINS, using_hsv=False)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals, nbins=N_HIST_BINS)
            feature = np.concatenate((rgb_hists, hsv_hists, nhists))
            labeled_features.append([feature, model_name])

    delete_model()

    features_filename = 'o{}_h{}_training_set.sav'.format(ORIENTATIONS_PER_OBJECT, N_HIST_BINS)

    pickle.dump(labeled_features, open(save_dir + features_filename, 'wb'))

