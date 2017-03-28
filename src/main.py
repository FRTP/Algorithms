#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def featureNormalize(X):
    """Returns a normalized version of X where the mean value of each feature
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


def sharpRatio(Ret):
    """Calculate sharpRatio.

    Parameters
    ----------
    Ret : list
        List of returns till time t.

    Returns
    -------
    float
        sharpRatio at time step t.
    """

    return np.mean(Ret) / np.std(Ret)


def rewardFunction(X, miu, delta, Ft, M):
    """Calculate reward at time step t.

    Parameters
    ----------
    X : list
    miu : float
        Maximum possible number of shares per transaction.
    delta : float
        The cost for a transaction at period t.

    Returns
    -------
    Ret : list
        List of returns till time t.
    sharpRatio(Ret) : list
        List of sharp ratios.
    """

    T = len(Ft) - 1

    Ret = miu * (Ft[1:T] * X[M+1:M+T] - delta * np.abs(Ft[2:T] - Ft[1:T-1]))

    return Ret, sharpRatio(Ret)


def traderFunction(X, w):
    return np.tanh(np.dot(X, w))


def costFunction(X, Xn, theta):
    miu = 1
    delta = 0.001

    M = len(theta) - 2
    T = len(X) - M

    # TODO: ...


def main():

    with open('retDAX.txt', 'r') as f:
        tmp = f.read()
    retDAX = list(map(lambda x: float(x), tmp.split()))

    with open('DAX.txt', 'r') as f:
        tmp = f.read()
    DAX = list(map(lambda x: float(x), tmp.split()))

    M = 10
    T = 500  # The number of time series inputs to the trader
    N = 100

    initial_theta = np.ones((M+2, 1))  # initialize theta

    X = retDAX

    Xn = featureNormalize(X)

    print(Xn)


if __name__ == '__main__':
    main()
