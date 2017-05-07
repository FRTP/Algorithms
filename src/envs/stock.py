import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import pandas as pd


class BalanceReward:
    def __init__(self, transaction_cost=0.0006):
        self.transaction_cost = transaction_cost
        self.balance = 0
        self.prices = None

    def update_balance(self, new_prices, action):
        self.balance += np.sum(action * (new_prices - self.prices)) - self.transaction_cost * np.abs(action)

    def __call__(self, new_prices, action):
        action -= 1
        if self.prices is None:
            self.prices = new_prices
            return 0
        old_balance = self.balance
        self.update_balance(new_prices, action)
        self.prices = new_prices

        return self.balance - old_balance


class SharpRatioReward(BalanceReward):
    pass


def read_prices(file_name):
    data = pd.read_csv(file_name)

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')
    data.reset_index(drop=True, inplace=True)

    prices = data['Open']

    return np.array(prices)


class StockEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, file_name, reward_type, use_twitter):
        self.prices = read_prices(file_name)
        self.price_diffs = np.concatenate(
            ([0], self.prices[1:] - self.prices[:-1]))

        if reward_type == 'balance':
            self.reward_class = BalanceReward
        elif reward_type == 'sharpe-ratio':
            self.reward_class = SharpRatioReward

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_dim = 5
        self.lim = np.ones((self.observation_dim,)) * 10000
        self.observation_space = spaces.Box(-self.lim, self.lim)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        self.cur_time += 1
        new_prices = self.prices[self.cur_time]

        state = self.price_diffs[
                self.cur_time: self.cur_time + self.observation_dim]
        reward = self.reward_calc(new_prices, action)
        done = bool(
            self.cur_time + self.observation_dim >= len(self.prices))

        return np.array(state), reward, done, {}

    def _reset(self):
        self.cur_time = np.random.randint(
            low=0, high=len(self.prices) - self.observation_dim)
        state = self.price_diffs[
                self.cur_time: self.cur_time + self.observation_dim]
        self.reward_calc = self.reward_class()

        return np.array(state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width,
                                           screen_height)

        budget_metric = np.abs(self.reward_calc.balance)

        budget_circle = rendering.make_circle(30 + budget_metric)
        if self.reward_calc.balance > 0:
            budget_circle.set_color(0, .5, 0)
        else:
            budget_circle.set_color(.5, 0, 0)

        budget_circle.add_attr(rendering.Transform(
            translation=(screen_width / 2, screen_height / 2)))

        self.viewer.add_onetime(budget_circle)

        return self.viewer.render(
            return_rgb_array=mode == 'rgb_array')
