import numpy as np
import matplotlib.pyplot as plt


def sharpRatio(Ret):
    """
    Calculate sharpRatio.

    Parameters
    ----------
    Ret : list
        List of returns till time t.

    Returns
    -------
    float
        sharpRatio at time step t.

    """
    if np.std(Ret) == 0:
        return 0
    else:
        return np.mean(Ret) / np.std(Ret)


def rewardFunction(X, miu, delta, Ft, M):
    """
    Calculate reward at time step t.

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
    """
    Represents the trading position at time t. Holdings at period t.
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
    """
    Update holdings at time steps t.

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
        # xt.extend(X[i-1:i+M-2])
        xt.extend(X[i:i+M])
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
    grad : list

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
        # xt.extend(Xn[i-1:i+M-2])
        xt.extend(Xn[i:i+M])
        xt.append(Ft[i-1])
        dFt[:, i] = (1 - traderFunction(xt, theta) ** 2) * \
                    (xt + theta[-1] * dFt[:, i-1])

    dRtFt = -1. * miu * delta * np.sign(Ft[1:]-Ft[:T])

    dRtFtt = miu * (X[M:M+T] + delta * np.sign(Ft[1:] - Ft[:T]))

    # % prefix = repmat(subs(subs(dSdA,a,A),b,B), T, 1) / T +
    #            subs(subs(dSdB,a,A),b,B) * 2*Ret/T;
    #
    # prefix = repmat((1/(- A^2 + B)^(1/2) + A^2/(B - A^2)^(3/2))/M, M, 1) +
    #          (-A/(2*(B - A^2)^(3/2))) * 2 * Ret / M

    A = np.sum(Ret) / T
    B = np.sum(Ret ** 2) / T

    prefix = np.tile((1 / np.sqrt(B - (A ** 2)) +
                     (A ** 2) / ((B - (A ** 2)) ** (3/2))) / M, (M, 1)) + \
                    (- A / (2 * (B - (A ** 2)) ** (3/2))) * 2 * Ret / M
    # print(prefix)
    # print(prefix.shape)

    # grad = np.sum(np.tile(prefix.T, (M+2, 1)).dot(
    #                 (np.tile(dRtFt.T, (M+2, 1))).dot(dFt[:, 1:]) +
    #                 np.tile(dRtFtt.T, (M+2, 1)).dot(dFt[:, :T]), 2))

    # print(np.tile(prefix.T, (M+2, 1)).shape)
    # print(np.tile(dRtFt.T, (M+2, 1)).dot(dFt[:, 1:].T).shape)
    # print(np.tile(dRtFtt.T, (M+2, 1)).dot(dFt[:, :T].T).shape)

    # TODO: fix this
    grad = np.sum(np.tile(prefix.T, (M+2, 1)).dot(
                    np.tile(dRtFt.T, (M+2, 1)).dot(dFt[:, 1:].T) +
                    np.tile(dRtFtt.T, (M+2, 1)).dot(dFt[:, :T].T)
                                                  ), axis=2)

    grad = -1 * grad

    return J, grad


if __name__ == '__main__':
    assert(0)
