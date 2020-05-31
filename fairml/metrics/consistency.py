import numpy as np


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
    df, target_variable, min_tau, max_tau, step_size,
):
    """Calculate consistency scores for the given taus in [min_tau,max_tau].

    Parameters
    ----------
    df : pd.DataFrame
    target_variable : str
    min_tau : float
    max_tau : float
    step_size : float
    
    Returns
    -------
    list
        Consistency scores for model and benchmark
    """

    model_consistency = []
    benchmark_consistency = []

    for tau in np.arange(min_tau, max_tau + step_size, step_size):

        model_col = "Y_" + str(int(tau * 100))
        benchmark_col = "Y_BENCHMARK_" + str(int(tau * 100))

        df[model_col] = 0
        df.loc[df.Y_SCORE >= tau, model_col] = 1

        df[benchmark_col] = 0
        df.loc[df.Y_BENCHMARK_SCORE >= tau, benchmark_col] = 1

        model_consistency_scores = df.apply(
            lambda row: _calculate_consistency(row, df, Y=model_col), axis=1
        )
        benchmark_consistency_scores = df.apply(
            lambda row: _calculate_consistency(row, df, Y=benchmark_col), axis=1
        )

        model_consistency.extend([1 - model_consistency_scores.mean()])
        benchmark_consistency.extend([1 - benchmark_consistency_scores.mean()])

    return np.array(model_consistency), np.array(benchmark_consistency)
