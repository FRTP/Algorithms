import sys

from gym.envs.registration import register
sys.path.insert(0, '../../Framework/src/')

register(
    id='FrontoPolarStocks-v0',
    entry_point='src.envs:StockEnv',
    kwargs={
        'reward_type': 'balance',
        'use_twitter': False
    },
    max_episode_steps=400,
    reward_threshold=300.0,
)
