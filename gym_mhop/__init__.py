from gym.envs.registration import register

register(
    id='mhop-v0',
    entry_point='gym_mhop.envs:MhopEnv'
)
