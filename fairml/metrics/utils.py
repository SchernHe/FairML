"""Provides useful functions to calculate fairness metrics"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import numpy as np
import pandas as pd


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


def _get_nn_idx(row, neigh, radius, columns):
    """Retrieve the NN of a sample within a specified radius.

    Parameters
    ----------
    row : pd.Series
    neigh : sklearn.NearestNeighbors
    radius : float
    columns : list

    Returns
    -------
    list
        Nearest Neighbors of given sample within radius
    """
    neigh_dist, neigh_idx = neigh.radius_neighbors([row[columns]], radius)
    return neigh_idx[0], len(neigh_idx[0])


def get_nn_idx(df, neigh, informative_variables, radius):
    """Assign each sample the indizes of NN.

    Parameters
    ----------
    df : pd.DataFrame
    neigh : sklearn.NearestNeighbors
    informative_variables : list
    radius : float

    Returns
    -------
    list
        Score values: Consistency, Accuracy and Precision
    """

    series = df.apply(
        lambda row: _get_nn_idx(row, neigh, radius, informative_variables), axis=1
    )

    df[["KNN_IDX", "Num_NN"]] = pd.DataFrame(series.tolist(), index=series.index)
    return df


def calculate_performance_scores(df, target_variable, min_tau, max_tau, step_size):

    accuracy_scores = []
    precision_scores = []

    for tau in np.arange(min_tau, max_tau + step_size, step_size):

        model_col = "Y_" + str(int(tau * 100))
        df[model_col] = df["Y_SCORE"].apply(lambda row: 1 if row >= tau else 0)

        accuracy_scores.extend([accuracy_score(df[target_variable], df[model_col])])
        precision_scores.extend([precision_score(df[target_variable], df[model_col])])

    return np.array(accuracy_scores), np.array(precision_scores)
