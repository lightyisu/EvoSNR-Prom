import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_metrics(preds, labels, positive_probs=None, include_extra=False):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    if not include_extra:
        return acc, precision, recall, f1, mcc

    positive_probs = preds if positive_probs is None else np.asarray(positive_probs)
    if np.unique(labels).size < 2:
        auprc = 0.0
    else:
        auprc = average_precision_score(labels, positive_probs)
    jaccard = jaccard_score(labels, preds, zero_division=0)
    return acc, precision, recall, f1, mcc, auprc, jaccard
