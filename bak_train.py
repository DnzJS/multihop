import os

import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
import settings


def train(opt):
    if settings.DEVICE == "cuda":
        print("cuda available")
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(
        opt.world, opt.stage, opt.action_type, opt.num_processes
    )
    model = PPO(envs.num_states, envs.num_actions)
    if settings.DEVICE == "cuda":
        model.cuda()
    model.share_memory()
    process = mp.Process(
        target=eval, args=(opt, model, envs.num_states, envs.num_actions)
    )
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if settings.DEVICE == "cuda":
        curr_states = curr_states.cuda()
    curr_episode = 0
    while True:
        # if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if settings.DEVICE == "cuda":
                [
                    agent_conn.send(("step", act))
                    for agent_conn, act in zip(envs.agent_conns, action.cpu())
                ]
            else:
                [
                    agent_conn.send(("step", act))
                    for agent_conn, act in zip(envs.agent_conns, action)
                ]

            state, reward, done, info = zip(
                *[agent_conn.recv() for agent_conn in envs.agent_conns]
            )
            state = torch.from_numpy(np.concatenate(state, 0))
            if settings.DEVICE == "cuda":
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        (
            _,
            next_value,
        ) = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = (
                gae
                + reward
                + opt.gamma * next_value.detach() * (1 - done)
                - value.detach()
            )
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                    int(
                        j * (opt.num_local_steps * opt.num_processes / opt.batch_size)
                    ) : int(
                        (j + 1)
                        * (opt.num_local_steps * opt.num_processes / opt.batch_size)
                    )
                ]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(
                    torch.min(
                        ratio * advantages[batch_indices],
                        torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon)
                        * advantages[batch_indices],
                    )
                )
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                print(f"loss: {total_loss.item()}")
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
