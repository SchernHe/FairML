import numpy as np
from sklearn.metrics import accuracy_score, precision_score


def _get_knn_idx(row, neigh, radius, columns):
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
    return neigh_idx[0]


def _calculate_consistency(row, df, Y):
    """Helper function to calculate the consistency score

    Parameters
    ----------
    row : pd.Series
    df : p.DataFrame
    Y : str
        Target Variable

    Returns
    -------
    pd.Series
        Consistency Score of each sample in the given dataframe
    """
    knn_idx = row["KNN_IDX"]
    knn_y = df.iloc[knn_idx][Y]
    row_y = row[Y]
    return np.abs(row_y - knn_y).mean()


def calculate_consistency(
    df,
    neigh,
    min_tau,
    max_tau,
    step_size,
    target_variable,
    informative_variables,
    radius,
):
    """Calculate consistency scores for the given taus in [min_tau,max_tau].

    Parameters
    ----------
    df : pd.DataFrame
    neigh : sklearn.NearestNeighbors
    min_tau : float
    max_tau : float
    step_size : float
    target_variable : str
    informative_variables : list
    radius : float

    Returns
    -------
    list
        Score values: Consistency, Accuracy and Precision
    """
    Y_tau = []
    consistency_scores = []
    accuracy_scores = []
    precision_scores = []

    for tau in np.arange(min_tau, max_tau + step_size, step_size):
        colname = "Y_" + str(int(tau * 100))
        df[colname] = df["Y_SCORE"].apply(lambda row: 1 if row >= tau else 0)
        Y_tau.extend([colname])

    df["KNN_IDX"] = df.apply(
        lambda row: _get_knn_idx(row, neigh, radius, informative_variables), axis=1
    )

    for Y in Y_tau:
        con_col = "Con_" + Y

        df[con_col] = df.apply(lambda row: _calculate_consistency(row, df, Y=Y), axis=1)

        consistency = 1 - df[con_col].mean()
        acc = accuracy_score(df[target_variable], df[Y])
        prec = precision_score(df[target_variable], df[Y])

        consistency_scores.extend([consistency])
        accuracy_scores.extend([acc])
        precision_scores.extend([prec])

    df["Benchmark_Con"] = df.apply(
        lambda row: _calculate_consistency(row, df, Y="Y_BENCHMARK"), axis=1
    )
    benchmark_consistency = 1 - df["Benchmark_Con"].mean()

    return consistency_scores, benchmark_consistency, accuracy_scores, precision_scores
