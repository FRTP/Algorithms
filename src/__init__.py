from gym.envs.registration import register

register(
    id='FrontoPolarStocks-v0',
    entry_point='src.envs:StockEnv',
    kwargs={
        'file_name': 'data/yah_stocks/EA.csv',
        'reward_type': 'balance',
        'use_twitter': False
    },
    max_episode_steps=400,
    reward_threshold=300.0,
)
