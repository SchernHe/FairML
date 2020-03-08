import numpy as np
from sklearn.neighbors import NearestNeighbors


def _calculate_consistency_for_each_item(df, row, Y, neigh,columns):
    """Helper function
    
    Parameters
    ----------
    row : pd.Series
    Y : str
        Prediction column name
    
    Returns
    -------
    float
        Consistency value of a given observation/row
    """
    neigh_dist, neigh_idx = neigh.kneighbors([row[columns]])
    return np.abs(row[Y] - df.iloc[neigh_idx[0]][Y].sum())


def calculate_consistency(df, Y, num_neigh):
    """Calculate consistency of  given dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
    Y  : str
        Prediction column name
    num_neigh : int
        Number of nearest neighbours
    
    Returns
    -------
    float
        Consistency of predictions in dataframe
    """
    columns = [col for col in df.columns if col != Y]
    neigh = NearestNeighbors(num_neigh)
    neigh.fit(df[columns])

    df["Consistency"] = df.apply(
        lambda row: _calculate_consistency_for_each_item(df, row, Y, neigh, columns), axis=1
    )

    consistency = 1 - 1 / (len(df) * num_neigh) * df["Consistency"].sum()
    df.drop("Consistency", axis=1, inplace=True)
    return consistency
