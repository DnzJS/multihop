import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
import time
from gym_mhop.envs import MhopEnv

import settings


def test():
    # ====== seed ========
    if settings.DEVICE == 'cuda':
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    env = MhopEnv()
    model = PPO(env.observation_space.shape[0], env.action_space.n)
    model.eval()
    state = env.reset()
    while True:
        if settings.DEVICE == 'cuda':
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        if done:
            print(reward)
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
