"""Provides useful functions to calculate fairness metrics"""
from sklearn.metrics import confusion_matrix
import numpy as np


def calculate_precision(df, target_variable, prediction_variable):
    """Calculate precision / positive predictive value PPV"""
    tn, fp, fn, tp = confusion_matrix(
        df[target_variable], df[prediction_variable]
    ).ravel()
    if (tp + fp) != 0:
        return (tp / (tp + fp)) * 100
    else:
        return np.nan


def calculate_recall(df, target_variable, prediction_variable):
    """Calculate recall / true positive rate TPR / sensitivity"""
    tn, fp, fn, tp = confusion_matrix(
        df[target_variable], df[prediction_variable]
    ).ravel()
    if (tp + fn) != 0:
        return (tp / (tp + fn)) * 100
    else:
        return np.nan


def calculate_fpr(df, target_variable, prediction_variable):
    """Calculate false positive rate FPR / false alarm ratio"""
    tn, fp, fn, tp = confusion_matrix(
        df[target_variable], df[prediction_variable]
    ).ravel()
    if (fp + tn) != 0:
        return (fp / (fp + tn)) * 100
    else:
        return np.nan
