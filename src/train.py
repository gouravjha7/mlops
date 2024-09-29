import os
import sys
import pickle

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


params = yaml.safe_load(open('params.yaml'))["train"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usages: \n")
    sys.stderr.write("\tpython train.py features Model\n")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]
seed = params['seed']
n_est = params["n_est"]
min_split = params["min_splits"]

with open(os.path.join(input, "train_1.pkl"), "rb") as fd:
    matrix = pickle.load(fd)

labels = matrix.iloc[:, 11].values
x = matrix.iloc[:, 1: 11].values
sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
sys.stderr.write("X matrix size {}\n".format(x.shape))
sys.stderr.write("Y matrix {}\n".format(labels.shape))

clf = RandomForestClassifier(
    n_estimators=n_est,
    min_samples_split=min_split,
    n_jobs=2,
    random_state=seed)

clf.fit(x, labels)

with open(output, "wb") as fd:
    pickle.dump(clf, fd)