import os
import json
import pickle
import sys
import traceback

import pandas as pd

from sklearn import tree
from sklearn.externals import joblib

from irisclassifier.resolve import paths


def read_config_file(config_json):
    """This function reads in a json file like hyperparameters.json or resourceconfig.json
    :param config_json: this is a string path to the location of the file (for both sagemaker or local)
    :return: a python dict is returned"""

    config_path = paths.config(config_json)
    if os.path.exists(config_path):
        json_data = open(config_path).read()
        return(json.loads(json_data))


def entry_point():
    """
    This function acts as the entry point for a docker container that an be used to train
    the model either locally or on Sagemaker depending in whichever context its called in as per resolve.paths class.

    """

    print('Starting the training.')
    try:
        hyper_params = read_config_file('hyperparameters.json')
        max_leaf_nodes = int(hyper_params['max_leaf_nodes'])


        train_data = pd.read_csv(paths.input(channel='training', filename='iris.csv'), header=None)

        # labels are in the first column
        train_y = train_data.ix[:,0]
        train_X = train_data.ix[:,1:]

        # Now use scikit-learn's decision tree classifier to train the model.
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        clf = clf.fit(train_X, train_y)

        # save the model
        with open(paths.model('decision-tree-model.pkl'), 'wb') as out:
            pickle.dump(clf, out)

            print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(paths.failure(), 'w') as s:
            s.write('Exception during training: ' + str(e) + str('\n') + trc)

        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == "__main__":
    entry_point()
