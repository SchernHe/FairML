import numpy as np
from sklearn.metrics import accuracy_score, precision_score


def _calculate_discrepancy(row, df, Y):
    """Helper function to calculate the discrepancy score

    Parameters
    ----------
    row : pd.Series
    df : p.DataFrame
    Y : str
        Target Variable

    Returns
    -------
    pd.Series
        discrepancy Score of each sample in the given dataframe
    """
    knn_idx = row["KNN_IDX"]
    knn_y = df.iloc[knn_idx][Y]
    n = len(knn_y)
    row_y = row[Y]
    num_of_deviations = np.abs(row_y - knn_y).sum()
    discrepancy = (n / (n + 10)) * num_of_deviations
    return discrepancy


def calculate_discrepancy(df, target_variable, Y_tau):
    """Calculate discrepancy scores for the given taus in [min_tau,max_tau].

    Parameters
    ----------
    df : pd.DataFrame
    target_variable : str
    Y_tau: list

    Returns
    -------
    list
        Score values: discrepancy, Accuracy and Precision
    """
    discrepancy_scores = []
    accuracy_scores = []
    precision_scores = []

    for Y in Y_tau:
        con_col = "Disc_" + Y

        df[con_col] = df.apply(lambda row: _calculate_discrepancy(row, df, Y=Y), axis=1)

        discrepancy = df[con_col].sum()
        acc = accuracy_score(df[target_variable], df[Y])
        prec = precision_score(df[target_variable], df[Y])

        discrepancy_scores.extend([discrepancy])
        accuracy_scores.extend([acc])
        precision_scores.extend([prec])

    df["Benchmark_Disc"] = df.apply(
        lambda row: _calculate_discrepancy(row, df, Y="Y_BENCHMARK"), axis=1
    )
    benchmark_discrepancy = df["Benchmark_Disc"].sum()

    return (
        discrepancy_scores,
        benchmark_discrepancy,
        accuracy_scores,
        precision_scores,
    )
