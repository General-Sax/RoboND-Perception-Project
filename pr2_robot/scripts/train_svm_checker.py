#!/usr/bin/env python
import pickle
import itertools
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import cross_validation
from sklearn import metrics

save_dir = './savefiles/'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fstr = '{}'
    if normalize:
        fstr = '{0:.2f}'
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fstr.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Create classifier
if __name__ == "__main__":
    print("\nTraining and selecting SVMs for all test data sets!")

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']

    # Load training data from disk
    # TODO: Add selection
    training_set = glob.glob(save_dir+'*_training_set.sav')[0]
    orientation_ct = (training_set.split('_')[0]).split('/')[-1]
    n_bins = training_set.split('_')[1]

    training_set = pickle.load(open(training_set, 'rb'))
    np.random.shuffle(training_set) # a pre-shuffle for good luck

    # Format the features and labels for use with scikit learn
    feature_list = []
    label_list = []

    for item in training_set:
        if np.isnan(item[0]).sum() < 1:
            feature_list.append(item[0])
            label_list.append(item[1])

    print('\nFeatures in Training Set: {}'.format(len(training_set)))
    print('Invalid Features in Training set: {}'.format(len(training_set)-len(feature_list)))

    # training_portion = 0.8
    # cutoff = int(round(training_portion * len(label_list)))

    X = np.array(feature_list)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    X_train = X_scaler.transform(X)
    # X_test = X_scaler.transform(X[cutoff:-1])

    y_train = np.array(label_list)
    # y_test = np.array(label_list[cutoff:-1])

    # Convert label strings to numerical encoding
    # train_encoder = LabelEncoder()
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    # y_test = encoder.fit_transform(y_test)

    results = {}

    for krnl in kernels:
      print("\n"+krnl)
      clf = svm.SVC(kernel=krnl)

      # Set up 5-fold cross-validation
      kf = cross_validation.KFold(len(X_train),
                                  n_folds=5,
                                  shuffle=True,
                                  random_state=1)

      # Perform cross-validation
      scores = cross_validation.cross_val_score(cv=kf,
                                               estimator=clf,
                                               X=X_train,
                                               y=y_train,
                                               scoring='accuracy'
                                              )
      print('Scores: ' + str(scores))
      print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), 2*scores.std()))

      # Gather predictions
      predictions = cross_validation.cross_val_predict(cv=kf,
                                                estimator=clf,
                                                X=X_train,
                                                y=y_train
                                               )

      accuracy_score = metrics.accuracy_score(y_train, predictions)
      results[accuracy_score] = krnl
      print('accuracy score: '+str(accuracy_score))

    krnl_final = results[sorted(results.keys())[-1]]
    print('\nbest kernel: '+krnl_final)

    clf = svm.SVC(kernel=krnl_final)

    # Set up 5-fold cross-validation
    kf = cross_validation.KFold(len(X_train),
                                n_folds=5,
                                shuffle=True,
                                random_state=1)

    predictions = cross_validation.cross_val_predict(cv=kf,
                                              estimator=clf,
                                              X=X_train,
                                              y=y_train
                                             )

    confusion_matrix = metrics.confusion_matrix(y_train, predictions)

    class_names = encoder.classes_.tolist()

    #Train the classifier
    clf.fit(X=X_train, y=y_train)

    model = {'classifier': clf, 'classes': encoder.classes_, 'scaler': X_scaler}

    # Save classifier to disk
    filename = 'fullmodel_{}_{}_{}.sav'.format(krnl_final, orientation_ct, n_bins)
    print "saving features to file: {}".format(filename)

    pickle.dump(model, open(save_dir+filename, 'wb'))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=encoder.classes_,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=encoder.classes_, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
