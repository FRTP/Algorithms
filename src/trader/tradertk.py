import numpy as np


def sharp_ratio(rewards):
    """
    Calculate sharp_ratio.

    Parameters
    ----------
    rewards : list
        List of rewards till time t.

    Returns
    -------
    float
        Sharpe ratio at time step t.

    """
    EPS = 1e-7

    return np.mean(rewards) / (np.std(rewards) + EPS)


def get_rewards(time_pos, returns, Ft, miu, delta):
    """
    Calculate rewards till time step `time_pos`.

    Parameters
    ----------
    time_pos : int
        Start-postion time step.
    returns : list
        List of returns till time step `time_pos`.
    Ft : list
        Holdings at time steps.
    miu : float
        Maximum possible number of shares per transaction.
    delta : float
        The cost for a transaction at period t.

    Returns
    -------
    ans : list
        List of Rewards till time t.
    s_ratio : float
        Sharp ratio at time step `time_pos`.

    """
    ans = [0]

    tmp = miu * (
        (Ft[:-1] * returns[1 + time_pos:len(Ft) + time_pos]) - (
            delta * abs(Ft[1:] - Ft[:-1])))

    ans.extend(tmp)

    s_ratio = sharp_ratio(ans)

    return ans, s_ratio


def build_x_matrix(time_pos, window_size, M, returns, w, Ft_prev):
    """
    Build matrix X of windows x_t. Size of every window is (M + 3).

    Parameters
    ----------
    time_pos : int
        Start-postion time step.
    window_size : int
        Size of historical window.
    M : int
        The number of time series inputs to the trader.
    returns : list
        List of returns.
    w : list
        Weights.
    Ft_prev : float

    Returns
    -------
    X : array of lists
        Array of windows x_t.

    """
    X = np.zeros((window_size, M + 3))

    X[0] = np.concatenate(
        ([1], list(reversed(returns[time_pos:1 + time_pos])),
         np.zeros(M), [Ft_prev]))

    for time_step in range(1, M + 1):
        X[time_step] = np.concatenate(([1], list(reversed(
            returns[time_pos:time_pos + time_step + 1])),
                                       np.zeros(M - time_step),
                                       # Padding with zeros.
                                       [trader_function(w, X[
                                           time_step - 1])]))
    # There is one difference. In the code above zeros are added.
    for time_step in range(M + 1, window_size):
        X[time_step] = np.concatenate(([1], list(reversed(
            returns[time_pos + time_step - M:
            time_pos + time_step + 1])),
                                       [trader_function(w, X[
                                           time_step - 1])]))
    return X


def build_x_vector(time_pos, M, returns, Ft_last):
    """
    Build an array - historical window x_t. Its length is (M + 3).

    Parameters
    ----------
    time_pos : int
        Time step to predict the agent's action in.
    M : int
        The number of time series inputs to the trader.
    returns : list
        List of returns.
    Ft_last : float
        Previois action of agent.

    Returns
    -------
    list
        Historical window x_t.

    """
    return np.concatenate(([1], list(reversed(
        returns[time_pos - M:time_pos + 1])), [Ft_last]))


def get_trader_func(X):
    """
    Get only last coordinates of vectors in matrix X.

    Parameters
    ----------
    X : list

    Returns
    -------
    list
        Values of trader function - the actions of agent.

    """
    return np.array(X)[:, -1]


def get_grad_F_w(X, w, dFdW_last):
    """
    Calculate gradient dF/dW.

    Parameters
    ----------
    X, w : list
    dFdW_last : list
        Gradient from last epoch.

    Returns
    -------
    dFdW : matrix
        Calculated gradient dF/dW.

    """

    def numeric_grad(x):
        d = 1e-5
        ans = np.zeros(w.shape)

        for i in range(len(w)):
            tmp = np.zeros(w.shape)
            tmp[i] = d
            ans[i] = (trader_function(x, w + tmp) -
                      trader_function(x, w - tmp)) / (2 * d)
        return ans

    dFdW = np.zeros(X.shape)
    numeric_dFdW = np.zeros(X.shape)
    # Define first dF/dW gradient of output matrix dFdW.
    dFdW[0] = (1 - (trader_function(X[0], w) ** 2)) * (
        X[0] + (w[-1] * dFdW_last))
    numeric_dFdW[0] = numeric_grad(X[0])
    for time_step in range(1, len(X)):
        dFdW[time_step] = (1 - (
            trader_function(X[time_step], w) ** 2)) * \
                          (X[time_step] + (
                              w[-1] * dFdW[time_step - 1]))
        numeric_dFdW[time_step] = numeric_grad(X[time_step])
    return numeric_dFdW, dFdW


def get_grad_S_w(time_pos, rewards, returns, Ft, miu, delta, dFdW, w,
                 Ft_first):
    """
    Calculate gradient dS/dW.

    Parameters
    ----------
    time_pos : int
    rewards : list
        List of Rewards.
    returns : list
        List of returns.
    Ft : list
        List of values of trader function.
    dFdW : list
        List of gradients dF/dW.
    miu, delta : float
    w : list
    Ft_first : float

    Returns
    -------
    grad : float
        Calculated gradient dS/dW.

    """
    A = sum(rewards) / len(rewards)
    B = sum(np.array(rewards) ** 2) / len(rewards)

    dSdA = (1 / np.sqrt(B - (A ** 2))) + (
        (A ** 2) / ((B - (A ** 2)) ** (3 / 2)))
    dSdB = -1 * (A / (2 * (((B - (A ** 2))) ** (3 / 2))))

    grad = 0.0

    # There are many points of code optimization and improvement.
    for t in range(1, len(rewards)):
        dAdR = 1 / len(rewards)
        dBdR = 2 * rewards[t] / len(rewards)

        dRdFt = -1 * miu * delta * np.sign(Ft[t] - Ft[t - 1])
        dRdFtt = (miu * returns[time_pos + t]) - dRdFt

        dFtdw = dFdW[t]
        dFttdw = dFdW[t - 1]

        grad += (dSdA * dAdR + dSdB * dBdR) * (
            dRdFt * dFtdw + dRdFtt * dFttdw)

    def numeric_grad():

        def help_func(w_tmp):
            X = build_x_matrix(time_pos, len(rewards),
                               len(dFdW[0]) - 3,
                               returns, w_tmp, Ft_first)
            F = get_trader_func(X)
            _, s_ratio = get_rewards(time_pos, returns, F, miu, delta)
            return s_ratio

        d = 1e-6
        ans = np.zeros(w.shape)
        for i in range(len(w)):
            tmp = np.zeros(w.shape)
            tmp[i] = d
            ans[i] = (help_func(w + tmp) - help_func(w - tmp)) / (
                2 * d)
        return ans

    return numeric_grad(), grad


def trader_function(X, w):
    """
    Represents the trading position at time t. Holdings at period t.
    Three types of positions that can be held: `long` (> 0),
    `short` (< 0), `neutral` (= 0).

    Parameters
    ----------
    X, w : list

    Returns
    -------
    float
        Value of trader_function.

    """
    return np.tanh(np.dot(X, w))


if __name__ == '__main__':
    assert 0
