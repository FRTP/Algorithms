import numpy as np
import matplotlib.pyplot as plt


#     with open('retDAX.txt', 'r') as f:
#         tmp = f.read()
#     retDAX = list(map(lambda x: float(x), tmp.split()))
#
#     with open('DAX.txt', 'r') as f:
#         tmp = f.read()
#     DAX = list(map(lambda x: float(x), tmp.split()))
#
#     M = 10
#     T = 500  # The number of time series inputs to the trader
#     N = 100
#
#     initial_theta = np.ones((M+2, 1))  # initialize theta
#
#     X = retDAX
#
#     Xn = featureNormalize(X)
#
#     print(Xn)


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
        List of returns at time steps t.
    miu : float
        Maximum possible number of shares per transaction.
    delta : float
        The cost for a transaction at period t.
    Ft : list
        Holdings at time steps t.
    M : int
        The number of time series inputs to the trader.

    Returns
    -------
    Ret : list
        List of returns till time t.
    sharpRatio(Ret) : float
        Sharp ratio at time step t.
    """

    T = len(Ft) - 1

    # The return at time t, considering the decision F_{t-1}
    Ret = miu * (Ft[:T] * X[M:M+T] - delta * np.abs(Ft[1:] - Ft[:T]))

    return Ret, sharpRatio(Ret)


def traderFunction(X, w):
    """Represents the trading position at time t. Holdings at period t.
    Three types of positions that can be held: `long` (> 0),
    `short` (< 0), `neutral` (= 0).

    Parameters
    ----------
    X, w : list

    Returns
    -------
    Value of traderFunction.
    """

    return np.tanh(np.dot(X, w))


def updateFt(X, theta, T):
    """Update holdings at time steps t.

    Parameters
    ----------
    X, theta : list
    T : int

    Returns
    -------
    Updated holdings at time steps t.
    """
    M = len(theta) - 2
    Ft = np.zeros(T+1)

    for i in range(1, T+1):
        xt = [1]
        xt.extend(X[i-1:i+m-2])
        xt.append(Ft[i-1])
        Ft[i] = traderFunction(xt, theta)

    return Ft


def costFunction(X, Xn, theta):
    """Calculate costFunction.

    Parameters
    ----------
    X, Xn, theta : list

    Returns
    -------
    J : float
    grad :

    """
    miu = 1
    delta = 0.001

    M = len(theta) - 2
    T = len(X) - M

    Ft = updateFt(Xn, theta, T)

    Ret, sharp = rewardFunction(X, miu, delta, Ft, M)

    J = sharp * -1

    dFt = np.zeros((M+2, T+1))
    for i in range(1, T+1):
        xt = [1]
        xt.extend(Xn[i-1:i+M-2])
        xt.append(Ft[i-1])
        dFt[:, i] = (1 - traderFunction(xt, theta) ** 2) * \
                    (xt + theta[M+2] * dFt[:, i-1])

    dRtFt = -1. * miu * delta * np.sign(Ft[1:]-Ft[:T])

    dRtFtt = miu * (X[M+1:T+M] + delta * np.sign(Ft[1:]-Ft[:T]))

    # TODO: prefix = ???
    # TODO: grad = ???


if __name__ == '__main__':
    assert(0)
