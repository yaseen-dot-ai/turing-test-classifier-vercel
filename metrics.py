import numpy as np
from collections import defaultdict

def confusion_matrix(y_true, y_pred, labels):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm

def per_class_pr(cm, labels):
    precision = {}
    recall = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        precision[label] = tp / max(cm[:, i].sum(), 1)
        recall[label] = tp / max(cm[i, :].sum(), 1)
    return precision, recall 