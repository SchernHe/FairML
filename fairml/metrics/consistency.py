import numpy as np
from sklearn.neighbors import NearestNeighbors


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
    neigh = NearestNeighbors(num_neigh,metric="euclidean")
    neigh.fit(df)
    return neigh


def _calculate_consistency(
    df, row, Y_Pred, neigh, columns, scaled=False, squared=False, radius=False
):
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
    if radius:
        neigh_dist, neigh_idx = neigh.radius_neighbors([row[columns]], radius)
    else:
        neigh_dist, neigh_idx = neigh.kneighbors([row[columns]])
    
    knn = df.iloc[neigh_idx[0]][Y_Pred].mean()
    e = 10 ** 10

    if scaled:
        knn = (1 - neigh_dist / (neigh_dist.sum() + e))[0] * knn

    if squared:
        return ((row[Y_Pred] - knn) ** 2).mean()
    else:
        return np.abs(row[Y_Pred] - knn)


def calculate_consistency(df, Y, neigh, scaled=False, squared=False, radius=False):
    """Calculate consistency of  given dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
    Y  : str
        Prediction column name

    
    Returns
    -------
    float
        Consistency value of a given observation/row, similar to Zemel et.al (2013)
    """

    columns = [col for col in df.columns if col != Y]

    df["Consistency"] = df.apply(
        lambda row: _calculate_consistency(df, row, Y, neigh, columns, scaled, squared,radius),
        axis=1,
    )
    consistency = 1 - df["Consistency"].mean()
    df.drop("Consistency", axis=1, inplace=True)
    return consistency
