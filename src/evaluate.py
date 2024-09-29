import sys
import os
import pickle
import numpy as np
import json
import math
import sklearn.metrics as metrics


if len(sys.argv) != 6:
    sys.stderr.write("Argument Error!\n")
    sys.stderr.write("\tpython evaluate.py model features scores prc roc")
    sys.exit(1)


model_file = sys.argv[1]
test_file = os.path.join(sys.argv[2], "test_1.pkl")
score_file = sys.argv[3]
prc_file = sys.argv[4]
roc_file = sys.argv[5]

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

with open(test_file, 'rb') as fd:
    matrix = pickle.load(fd)

x = matrix.iloc[:, 1:11].values

clean_x = np.where(np.isnan(x), 0, x)
labels_pred = model.predict(clean_x)

prediction_by_class = model.predict_proba(clean_x)
predictions = prediction_by_class[:, 1]
print(predictions)

precision, recall, prc_thres = metrics.precision_recall_curve(labels_pred,
                                                              predictions,
                                                              pos_label=1)
fpr, tpr, roc_thres = metrics.roc_curve(labels_pred,
                                        predictions,
                                        pos_label=1)
avg_prec = metrics.average_precision_score(labels_pred, predictions)
roc_auc = metrics.roc_auc_score(labels_pred, predictions)

with open(score_file, "w") as fd:
    json.dump({"avg_pred": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

nth_point = math.ceil(len(prc_thres) / 1000)
prc_points = list(zip(precision, recall, prc_thres))[::nth_point]


with open(prc_file, "w") as fd:
    json.dump({
        "prc": [
            {
                "precision": p,
                "recall": r,
                "threshold": t
            }
            for p, r, t in prc_points
        ]
    }, fd, indent=4
    )

with open(roc_file, "w") as fd:
    json.dump(
        {
            "roc": [
                {"fpr": fp,
                 "tpr": tp,
                 "threshold": t}
                for fp, tp, t in zip(fpr, tpr, roc_thres)
            ]
        }, fd, indent=4
    )
