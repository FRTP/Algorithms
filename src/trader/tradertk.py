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
        sharp_ratio at time step t.

    """

    EPS = 1e-5

    if np.std(rewards) == 0:
        return 0
    else:
        return np.mean(rewards) / (np.std(rewards) + EPS)


def get_rewards(t1_input, r_input, F_input, miu_input, delta_input):
    """
    Calculate rewards till time step t_input.

    Parameters
    ----------
    t1_input : int
        Start-postion time step.
    r_input : list
        List of returns till time step t_input.
    F_input : list
        Holdings at time steps.
    miu_input : float
        Maximum possible number of shares per transaction.
    delta_input : float
        The cost for a transaction at period t.

    Returns
    -------
    ans : list
        List of Rewards till time t.
    s_ratio : float
        Sharp ratio at time step t_input.

    """
    ans = [0]

    tmp = miu_input * ((F_input[:-1] * r_input[1+t1_input:len(F_input) +
                        t1_input]) - (delta_input * abs(F_input[1:] -
                                                        F_input[:-1])))

    ans.extend(tmp)

    s_ratio = sharp_ratio(ans)

    return ans, s_ratio


def build_x_matrix(t1_input, maxT_input, M_input,
                   r_input, w_input, F_t_1_input):
    """
    Build matrix X of windows x_t. Size of every window is (M_input + 3).

    Parameters
    ----------
    t1_input : int
        Start-postion time step.
    maxT_input : int
        Maximum time step.
    M_input : int
        The number of time series inputs to the trader.
    r_input : list
        List of returns.
    w_input : list
        Weights.
    F_t_1_input : float

    Returns
    -------
    X_ans : array of lists
        Array of windows x_t.

    """
    X_ans = np.zeros((maxT_input, M_input + 3))

    X_ans[0] = np.concatenate(([1], list(reversed(
               r_input[t1_input:1 + t1_input])), np.zeros(M_input),
                                                 [F_t_1_input]))

    for time_step in range(1, M_input + 1):
        X_ans[time_step] = np.concatenate(([1], list(reversed(
                           r_input[t1_input:t1_input + time_step + 1])),
                           np.zeros(M_input - time_step),
                           [trader_function(w_input, X_ans[time_step - 1])]))
    # There is one difference. In the code above zeros are added.
    for time_step in range(M_input + 1, maxT_input):
        X_ans[time_step] = np.concatenate(([1], list(reversed(
                           r_input[t1_input+time_step-M_input:
                                   t1_input+time_step+1])),
                           [trader_function(w_input, X_ans[time_step - 1])]))
    return X_ans


def build_x_vector(t_pred_input, M_input, r_input, F_last_input):
    """
    Build an array - historical window x_t. Its length is (M_input + 3).

    Parameters
    ----------
    t_pred_input : int
        Time step to predict the agent's action in.
    M_input : int
        The number of time series inputs to the trader.
    r_input : list
        List of returns.
    F_last_input : float
        Previois action of agent.

    Returns
    -------
    list
        Historical window x_t.

    """
    return np.concatenate(([1], list(reversed(r_input[t_pred_input -
                          M_input:t_pred_input + 1])), [F_last_input]))


def get_trader_func(X_input):
    """
    Get only last coordinates of vectors in matrix X_input.

    Parameters
    ----------
    X_input : list

    Returns
    -------
    list
        Values of trader function - the actions of agent.

    """
    return np.array(X_input)[:, -1]


def get_grad_F_w(X_input, w_input, dFdW_last_values):
    """
    Calculate gradient dF/dW.

    Parameters
    ----------
    X_input, w_input : list
    dFdW_last_values : list
        Gradient from last epoch.

    Returns
    -------
    dFdW : matrix
        Calculated gradient dF/dW.

    """
    def numeric_grad(x, w):
        delta = 1e-5
        ans = np.zeros(w.shape)

        for i in range(len(w)):
            tmp = np.zeros(w.shape)
            tmp[i] = delta
            ans[i] = (trader_function(x, w + tmp) -
                      trader_function(x, w - tmp)) / (2 * delta)

        return ans

    dFdW = np.zeros(X_input.shape)
    # Define first dF/dW gradient of output matrix dFdW.
    dFdW[0] = (1 - (trader_function(X_input[0], w_input) ** 2)) * \
              (X_input[0] + (w_input[-1] * dFdW_last_values))
    for time_step in range(1, len(X_input)):
        dFdW[time_step] = (1 - (trader_function(X_input[time_step],
                                                w_input) ** 2)) * \
                    (X_input[time_step] + (w_input[-1] * dFdW[time_step-1]))
        # print(max(abs(dFdW[time_step] - numeric_grad(X_input[time_step],
        #                                              w_input))))
    return dFdW


def get_grad_S_w(t1_input, rewards_input, r_input, F_input, miu_input,
                 delta_input, dFdW_input, w_input, F_first_value_input):
    """
    Calculate gradient dS/dW.

    Parameters
    ----------
    t1_input : int
    rewards_input : list
        List of Rewards.
    r_input : list
        List of returns.
    F_input : list
        List of values of trader function.
    dFdW_input : list
        List of gradients dF/dW.
    miu_input, delta_input : float
    w_input : list

    Returns
    -------
    grad : float
        Calculated gradient dS/dW.

    """
    A = sum(rewards_input) / len(rewards_input)
    B = sum(np.array(rewards_input) ** 2) / len(rewards_input)

    dSdA = (1 / np.sqrt(B - (A ** 2))) + ((A ** 2) / ((B - (A ** 2)) ** (3/2)))
    dSdB = -1 * (A / (2 * (((B - (A ** 2))) ** (3/2))))

    grad = 0.0

    # There are many points of code optimization and improvement.
    for t in range(1, len(rewards_input)):
        dAdR = 1 / len(rewards_input)
        dBdR = 2 * rewards_input[t] / len(rewards_input)

        dRdFt = -1 * miu_input * delta_input * np.sign(F_input[t] -
                                                       F_input[t-1])
        dRdFtt = (miu_input * r_input[t1_input+t]) - dRdFt

        dFtdw = dFdW_input[t]
        dFttdw = dFdW_input[t-1]

        grad += (dSdA*dAdR + dSdB*dBdR) * (dRdFt*dFtdw + dRdFtt*dFttdw)

    def numeric_grad():

        def help_func(w):
            X = build_x_matrix(t1_input, len(rewards_input), len(dFdW_input[0])
                               - 3, r_input, w, F_first_value_input)
            F = get_trader_func(X)
            rewards, s_ratio = get_rewards(t1_input, r_input, F,
                                           miu_input, delta_input)
            return s_ratio

        delta = 1e-10
        ans = np.zeros(w_input.shape)
        for i in range(len(w_input)):
            tmp = np.zeros(w_input.shape)
            tmp[i] = delta
            ans[i] = (help_func(w_input + tmp) -
                      help_func(w_input - tmp)) / (2 * delta)
        return ans

    # print(max(abs(numeric_grad() - grad)))

    return grad


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
    assert(0)
