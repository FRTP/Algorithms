import numpy as np


class Trader:
    """
    Trader class.
    Start trading using initial parameters.

    Parameters
    ----------
    just_data : list
        List of prices.
    returns : list
        List of returns.
    M : int
        The number of time series inputs to the trader.
    window_size : int
        Size of historical window.
    n_steps : int
        Number of steps to make predictions.
    start_position : int
        Start-postion time step.
    name : str
        The name of Stocks (the default is '_').
    miu : float, optinonal
        Maximum possible number of shares per trasaction (the default is 1).
    delta : float, optinonal
        The cost for a transaction at period t (the default is 0.0006, which is
        used at stocks in Moscow).
    num_iter_learn : int, optinonal
        (the default is 0.01).
    learning_rate_0 : float, optinonal
        Learning rate (the default is 0.01).
    is_dynam_lr : bool, optional
        Flag to use or not dynamic learning rate (the default is False).
    debug : bool, optional
        Flag to calculate or not numerical grads (the default is False).

    Returns
    -------
    Nothing. It just trades.

    """
    def __init__(self, returns, M, window_size, n_steps, start_position,
                 name='_', miu=1, delta=0.0006, num_iter_learn=100,
                 learning_rate_0=0.01, is_dynam_lr=False, debug=False):

        self.returns = returns
        self.M = M
        self.window_size = window_size
        self.n_steps = n_steps
        self.start_position = start_position

        self.name = name
        self.miu = miu
        self.delta = delta
        self.num_iter_learn = num_iter_learn
        self.learning_rate_0 = learning_rate_0
        self.is_dynam_lr = is_dynam_lr

        self.w = np.random.uniform(-0.1, 0.1, M + 3)  # Weights initialization.
        self.F_predictions = []  # Agent's actions.

        # They are both needed for training.
        self.F_last_value = 0
        self.F_first_value = 0
        self.dFdW_last_values = np.zeros(self.w.shape)
        # Use `self.t1` as iterator.
        self.t1 = self.start_position

    def one_step():


    def train_on_history():
        # Training on previous history.
        for _ in range(self.num_iter_learn):
            X = build_x_matrix(self.t1, self.window_size, self.M, returns, w, F_first_value)

            # Just getting last coordinate of vectors from matrix X.
            F = get_trader_func(X)

            # Calculate rewards and sharpe.
            Rewards, s_ratio = get_rewards(t1, returns, F, miu, delta)

            # Calculate gradient dF/dW.
            numeric_dFdW, dFdW = get_grad_F_w(X, w, dFdW_last_values)

            # Calculate gradient dS/dW.
            numeric_grad, grad = get_grad_S_w(t1, Rewards, returns, F, miu, delta, dFdW, w, F_first_value)

            # Update weights.
            w += learning_rate * grad
        ## End of the training on history.

    def get_weights(self):
        """
        Returns
        -------
        w : list
            Weights.

        """
        return self.w

    def sharp_ratio(self, rewards):
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

    def get_rewards(self, time_pos, returns, Ft):
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

        Returns
        -------
        ans : list
            List of Rewards till time t.
        s_ratio : float
            Sharp ratio at time step `time_pos`.

        """
        ans = [0]

        tmp = self.miu * ((Ft[:-1] * returns[1 + time_pos:len(Ft) + time_pos])
                          - (self.delta * abs(Ft[1:] - Ft[:-1])))

        ans.extend(tmp)

        s_ratio = sharp_ratio(ans)

        return ans, s_ratio

    def build_x_matrix(self, time_pos, returns, Ft_prev):
        """
        Build matrix X of windows x_t. Size of every window is (M + 3).

        Parameters
        ----------
        time_pos : int
            Start-postion time step.
        returns : list
            List of returns.
        Ft_prev : float

        Returns
        -------
        X : array of lists
            Array of windows x_t.

        """
        X = np.zeros((window_size, self.M + 3))

        X[0] = np.concatenate(([1], list(reversed(returns[time_pos:1 +
                               time_pos])), np.zeros(self.M), [Ft_prev]))

        for time_step in range(1, self.M + 1):
            X[time_step] = np.concatenate(([1], list(reversed(
                               returns[time_pos:time_pos + time_step + 1])),
                               np.zeros(self.M - time_step),  # Padding with
                                                              # zeros.
                               [trader_function(self.w, X[time_step - 1])]))
        # There is one difference. In the code above zeros are added.
        for time_step in range(self.M + 1, self.window_size):
            X[time_step] = np.concatenate(([1], list(reversed(
                               returns[time_pos + time_step - self.M:
                                       time_pos + time_step + 1])),
                               [trader_function(w, X[time_step - 1])]))
        return X

    def build_x_vector(self, time_pos, returns, Ft_last):
        """
        Build an array - historical window x_t. Its length is (M + 3).

        Parameters
        ----------
        time_pos : int
            Time step to predict the agent's action in.
        returns : list
            List of returns.
        Ft_last : float
            Previois action of agent.

        Returns
        -------
        list
            Historical window x_t.

        """
        return np.concatenate(([1], list(reversed(returns[time_pos - self.M:
                                                  time_pos + 1])), [Ft_last]))

    def get_trader_func(self, X):
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

    def get_grad_F_w(self, X, dFdW_last):
        """
        Calculate gradient dF/dW.

        Parameters
        ----------
        X: list
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
                ans[i] = (trader_function(x, self.w + tmp) -
                          trader_function(x, self.w - tmp)) / (2 * d)
            return ans

        dFdW = np.zeros(X.shape)
        # Define first dF/dW gradient of output matrix dFdW.
        dFdW[0] = (1 - (trader_function(X[0], self.w) ** 2)) * (X[0] +
                                                                (self.w[-1] *
                                                                dFdW_last))
        if self.debug:
            numeric_dFdW = np.zeros(X.shape)
            numeric_dFdW[0] = numeric_grad(X[0])
            for time_step in range(1, len(X)):
                dFdW[time_step] = (1 - (trader_function(X[time_step], self.w)
                                   ** 2)) * (X[time_step] + (self.w[-1] *
                                             dFdW[time_step-1]))
                numeric_dFdW[time_step] = numeric_grad(X[time_step])
            return numeric_dFdW, dFdW
        else:
            return [], dFdW

    def get_grad_S_w(time_pos, rewards, returns, Ft, dFdW, Ft_first):
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
        Ft_first : float

        Returns
        -------
        grad : float
            Calculated gradient dS/dW.

        """
        A = sum(rewards) / len(rewards)
        B = sum(np.array(rewards) ** 2) / len(rewards)

        dSdA = (1 / np.sqrt(B - (A ** 2))) + \
               ((A ** 2) / ((B - (A ** 2)) ** (3/2)))
        dSdB = -1 * (A / (2 * (((B - (A ** 2))) ** (3/2))))

        grad = 0.0

        # There are many points of code optimization and improvement.
        for t in range(1, len(rewards)):
            dAdR = 1 / len(rewards)
            dBdR = 2 * rewards[t] / len(rewards)

            dRdFt = -1 * miu * delta * np.sign(Ft[t] - Ft[t-1])
            dRdFtt = (miu * returns[time_pos+t]) - dRdFt

            dFtdw = dFdW[t]
            dFttdw = dFdW[t-1]

            grad += (dSdA*dAdR + dSdB*dBdR) * (dRdFt*dFtdw + dRdFtt*dFttdw)

        def numeric_grad():

            def help_func():
                X = build_x_matrix(time_pos, len(rewards), len(dFdW[0]) - 3,
                                   returns, Ft_first)
                F = get_trader_func(X)
                _, s_ratio = get_rewards(time_pos, returns, F, self.miu,
                                         self.delta)
                return s_ratio

            d = 1e-6
            ans = np.zeros(self.w.shape)
            for i in range(len(self.w)):
                tmp = np.zeros(self.w.shape)
                tmp[i] = d
                ans[i] = (help_func(self.w + tmp) - help_func(self.w - tmp)) /\
                         (2 * d)
            return ans

        if self.debug:
            return numeric_grad(), grad
        else:
            return [], grad

    def trader_function(X):
        """
        Represents the trading position at time t. Holdings at period t.
        Three types of positions that can be held: `long` (> 0),
        `short` (< 0), `neutral` (= 0).

        Parameters
        ----------
        X : list

        Returns
        -------
        float
            Value of trader_function.

        """
        return np.tanh(np.dot(X, self.w))
