import numpy as np
from sklearn.neighbors import NearestNeighbors


def _calculate_consistency_for_each_item(df, row, Y, neigh, columns):
    """Helper function
    
    Parameters
    ----------
    row : pd.Series
    Y : str
        Prediction column name
    
    Returns
    -------
    float
        Consistency value of a given observation/row (Zemel et.al 2013)

        Consitency_Row= | Y_Row -  SUM_KNN(Y_KNN) |
    """
    neigh_dist, neigh_idx = neigh.kneighbors([row[columns]])
    return np.abs(row[Y] - df.iloc[neigh_idx[0]][Y]).mean()


def fit_nearest_neighbors(df, num_neigh):
    """ Helper function 
    Parameters
    ----------
        num_neigh : int
        Number of nearest neighbours

    Returns
    -------
        neigh: Fitted NN
        """
    neigh = NearestNeighbors(num_neigh)
    neigh.fit(df)
    return neigh


def calculate_consistency(df, Y, neigh):
    """Calculate consistency of  given dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
    Y  : str
        Prediction column name

    
    Returns
    -------
    float
        Consistency of predictions in dataframe (Zemel et.al 2013)

        Consistency = 1 - (1/n) * SUM_ROW(Consitency_Row)
    """
    columns = [col for col in df.columns if col != Y]

    df["Consistency"] = df.apply(
        lambda row: _calculate_consistency_for_each_item(df, row, Y, neigh, columns),
        axis=1,
    )
    consistency = 1 - df["Consistency"].mean()
    df.drop("Consistency", axis=1, inplace=True)
    return consistency
