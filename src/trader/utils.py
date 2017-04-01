import numpy as np


def featureNormalize(X):
    """
    Returns a normalized version of X where the mean value of each feature
    is 0 and the standard deviation is 1. This is often a good preprocessing
    step to do when working with learning algorithms.

    Parameters
    ----------
    X : list

    Returns
    -------
    X_norm : list
        Normalized version of X.

    """
    m = len(X)
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu) / sigma

    return X_norm
