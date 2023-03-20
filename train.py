from models import PolicyNN
import gym
from gym.vector import VectorEnv
from gym_mhop.envs import MhopEnv
from GlobalConfig import GlobalConfig as gconf
import numpy as np
import torch
import time
import sys, os
import pickle
import math
import shutil
from torch.distributions.categorical import Categorical
import itertools
from cuda_bf import CUDA_BF_Solver

import inspect


class RouteRecord:
    def __init__(self):
        self.map_id = -1
        self.route = None
        self.value = 0
        self.eps = 0

    def update(self, map_id, eps, value, route):
        self.map_id = map_id
        self.eps = eps
        self.value = value
        self.route = route

    def __str__(self):
        return f"map_id: {self.map_id} eps: {self.eps} value: {self.value} route: {self.route}"


class Trainer:
    """
    Working progress of the training. Only one copy per experiment.
    """

    def __init__(self):
        self.model = None
        self.envs = None

        self.total_eps = 0  # total training eps for the model
        self.eps = 0  # epochs trained for the current map
        self.map_id = -1  # current map id
        self.map = None
        self.total_maps = 0  # total number of maps seem

        self.train_best = RouteRecord()
        self.test_best = RouteRecord()

        self.his_train_avg = []

        # self.epr_train = (
        #    []
        # )  # record the (average, if train on multiple envs) epoch-reward history
        ## format: [ep, avg_reward]
        # self.epr_test = []  # record the average reward of the model, predict X times
        ## format: [ep, avg_reward]

        ## deterministic model selection reward
        ## format: [ep, reward, solution]
        # self.epr_deterministic = []

        ## record the best N solutions of the current map (from training)
        ## format: [ep, reward, solution]
        # self.best_train = []

        ## record the best N solutions of the current map (from testing)
        ## format: [ep, reward, solution]
        # self.best_test = []

    def lb_sort_func(item):
        """
        leader board list sorting function
        """
        return item[0]

    def new_map(self):
        self.map_id += 1

        _map = MhopEnv.generate_map()

        # save map
        if not os.path.exists("outputs/maps"):
            os.makedirs("outputs/maps")
        with open(f"outputs/maps/m{self.map_id}", "wb") as f:
            pickle.dump(_map, f)

        self._map = _map
        self.search_optimal()

    def search_optimal(self):
        solver = CUDA_BF_Solver()
        solver.assign_map(self._map.gen_parms, self._map)
        solver.solve(hops=3)

    def load_map(self):
        with open(f"outputs/maps/m{self.map_id}", "rb") as f:
            self._map = pickle.load(f)

    def expose_all_trajectories(self):
        num_envs = self.num_envs

        m_obs = np.zeros((num_envs, gconf.TOTAL_HOPS, self.OBS_SHAPE))
        m_done = np.full((num_envs, gconf.TOTAL_HOPS, 1), fill_value=False)
        m_rs = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        m_vs = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        # one-hot encoding
        m_ats = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        m_ds = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        m_logprobs = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))

        M = self._map.M
        observations, infos = self.envs.reset()
        all_permu = np.array(list(itertools.permutations(range(M), 2)), dtype=int) + 1
        # terminate signal
        t_signal = np.zeros((num_envs, 1), dtype=int)
        all_permu = np.concatenate((all_permu, t_signal), axis=1)

        for hop in range(gconf.TOTAL_HOPS):
            actions = all_permu[:, hop]
            observations_new, rewards, dones, truncates, infos = self.envs.step(actions)

            tmp_obs_torch = torch.from_numpy(observations)
            tmp_obs_torch = tmp_obs_torch.float()
            with torch.no_grad():
                logits, values = self.model(tmp_obs_torch)
                m_vs[:, hop, 0] = values.flatten()
            probs = Categorical(logits=logits)

            # calculate log probs
            logprobs = probs.log_prob(torch.tensor(actions))

            m_obs[:, hop, :] = np.array(observations)
            m_rs[:, hop, 0] = np.array(rewards)
            m_ds[:, hop, 0] = np.array(dones)
            m_ats[:, hop, 0] = np.array(actions)
            m_logprobs[:, hop, 0] = np.array(logprobs)

            observations = observations_new

        self.envs.close()

        data = (m_obs, m_ats, m_rs, m_ds, m_vs, m_logprobs)
        #print(np.max(m_rs[:, 2, 0]))
        argmx = np.argmax(m_rs[:, 2, 0])
        print(m_ats[argmx, :, 0], self._map.furthest_pair_ls, np.max(m_rs[:, 2, 0]))
        return data


    def interact_fixed_hops(self, hops):
        # envs: VectorEnv, model: PolicyNN, seed=None
        # init memory
        num_envs = gconf.TOTAL_ENVS
        m_obs = np.zeros((num_envs, gconf.TOTAL_HOPS, self.OBS_SHAPE))
        m_done = np.full((num_envs, gconf.TOTAL_HOPS, 1), fill_value=False)
        m_rs = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        m_vs = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        # one-hot encoding
        m_ats = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        m_ds = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))
        m_logprobs = np.zeros((num_envs, gconf.TOTAL_HOPS, 1))

        # obs = []
        # ds = []
        # rs = []
        # ats = []

        observations = None
        # interact and aquire data
        for hop in range(hops):
            if hop == 0:
                observations, infos = self.envs.reset()

            tmp_obs_torch = torch.from_numpy(observations)
            tmp_obs_torch = tmp_obs_torch.float()

            # a flag provided to the masking function
            # the last action always send the terminate signal
            t_flag = hop < (hops - 1)

            with torch.no_grad():
                logits, values = self.model(tmp_obs_torch)
                m_vs[:, hop, 0] = values.flatten()

            probs = Categorical(logits=logits)
            # mask actions
            self.mask_actions(logits, non_terminate=t_flag)
            actions = PolicyNN.sample_action(logits, deterministic=False)
            observations_new, rewards, dones, truncates, infos = self.envs.step(actions)

            # calculate log probs
            logprobs = probs.log_prob(torch.tensor(actions))

            m_obs[:, hop, :] = np.array(observations)
            m_rs[:, hop, 0] = np.array(rewards)
            m_ds[:, hop, 0] = np.array(dones)
            m_ats[:, hop, 0] = np.array(actions)
            m_logprobs[:, hop, 0] = np.array(logprobs)

            observations = observations_new

        self.envs.close()

        # ====record history====
        # update training best
        b_value = np.max(m_rs[:, -1, 0])
        b_index = np.argmax(m_rs[:, -1, 0])
        # b_route = list(infos["final_info"][b_index]["route"]) + [
        #    self._map.furthest_pair_ls[1]
        # ]
        b_route = None

        if b_value > self.train_best.value:
            self.train_best.update(self.map_id, self.eps, b_value, b_route)
            print(f"update {self.eps} {b_value} {b_route}")

        # training avg
        avg = np.average(m_rs[:, -1, 0])
        mx = np.max(m_rs[:, -1, 0])
        occ = np.count_nonzero(m_rs == mx)
        self.his_train_avg.append((self.eps, avg))
        if self.eps % 10 == 0:
            print(self.eps, avg, mx, occ, sep="\t\t")
            if avg == 0:
                breakpoint()
        # ======================

        data = (m_obs, m_ats, m_rs, m_ds, m_vs, m_logprobs)
        return data

    def mask_actions(self, logits, non_terminate):
        if non_terminate == False:
            logits[:, 1:] = -math.inf
            logits[:, 0] = 1.0
            return

        # get nodes to mask
        nstm = self.envs.call("get_nodes_to_mask")
        # mask nodes
        for i in range(gconf.TOTAL_ENVS):
            # mask terminate signal
            logits[i, 0] = -math.inf

            # mask selected nodes and sender/receiver nodes
            for j in range(len(nstm[i])):
                mask_node = nstm[i][j]
                logits[i, mask_node] = -math.inf

            # mask nodes that not exists at all
            for j in range(self._map.M, gconf.MAX_M + 1):
                logits[i, j] = -math.inf

    def update_weights(self, data):
        (m_obs, m_ats, m_rs, m_ds, m_vs, m_logprobs) = data
        m_obs = torch.from_numpy(m_obs)
        m_obs = m_obs.type(torch.float)
        m_ats = torch.from_numpy(m_ats)
        # m_rs shape: rewards, [ENVS, hops, 1]
        m_rs = torch.from_numpy(m_rs)
        m_ds = torch.from_numpy(m_ds)
        m_vs = torch.from_numpy(m_vs)
        m_adv = torch.zeros_like(m_rs)  # .to(gconf.DEVICE)
        m_logprobs = torch.from_numpy(m_logprobs)

        # bootstrap value if not done
        with torch.no_grad():
            # next_value = self.model.get_value(next_obs).reshape(1, -1)
            # garentee every last step is the terminal
            next_value = 0
            next_done = 1.0
            if gconf.USE_GAE:
                lastgaelam = 0
                for t in reversed(range(gconf.TOTAL_HOPS)):
                    if t == gconf.TOTAL_HOPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - m_ds[:, t + 1, 0]
                        nextvalues = m_vs[:, t + 1, 0]
                    delta = (
                        m_rs[:, t, 0]
                        + gconf.gamma * nextvalues * nextnonterminal
                        - m_vs[:, t, 0]
                    )
                    m_adv[:, t, 0] = lastgaelam = (
                        delta
                        + gconf.gamma * gconf.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = m_adv + m_vs
            else:
                returns = torch.zeros_like(m_rs).to(gconf.DEVICE)
                for t in reversed(range(gconf.TOTAL_HOPS)):
                    if t == gconf.TOTAL_HOPS - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - m_ds[:, t + 1, 0]
                        next_return = returns[t + 1]
                    returns[t] = (
                        m_rs[:, t, 0] + gconf.gamma * nextnonterminal * next_return
                    )
                m_adv = returns - m_vs

        # concatenate envs vector data
        def reshape_data(x):
            return x.view(x.shape[0] * x.shape[1], x.shape[2])

        m_obs = reshape_data(m_obs)
        m_ats = reshape_data(m_ats)
        m_ds = reshape_data(m_ds)
        m_vs = reshape_data(m_vs)
        m_rs = reshape_data(m_rs)
        m_adv = reshape_data(m_adv)
        m_logprobs = reshape_data(m_logprobs)

        # Optimizing the policy and value network
        # TODO: multiple runs?
        batch_size = self.num_envs * gconf.TOTAL_HOPS * gconf.ENV_RUNS
        minibatch_size = batch_size // gconf.num_minibatches
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(gconf.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                newlogits, newvalue = self.model(m_obs[mb_inds])

                newprob = Categorical(logits=newlogits)
                newlogprob = newprob.log_prob(m_ats[mb_inds])
                entropy = newprob.entropy()

                logratio = newlogprob - m_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > gconf.clip_coef).float().mean().item()
                    ]

                mb_advantages = m_adv[mb_inds]
                if gconf.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - gconf.clip_coef, 1 + gconf.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if gconf.clip_vloss:
                    v_loss_unclipped = (newvalue - m_rs[mb_inds]) ** 2
                    v_clipped = m_vs[mb_inds] + torch.clamp(
                        newvalue - m_vs[mb_inds],
                        -gconf.clip_coef,
                        gconf.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - m_rs[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - m_rs[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - gconf.ent_coef * entropy_loss + v_loss * gconf.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gconf.max_grad_norm
                )
                self.optimizer.step()

    def one_epoch(self, load_model=None):
        # enter train loop:
        # take several steps
        # save observations, reward, done

        # run multiple times to gather enough training samples
        obs = None
        ats = None
        rs = None
        ds = None
        for i in range(gconf.ENV_RUNS):
           #tmp = self.interact_fixed_hops(gconf.TOTAL_HOPS)
           tmp = self.expose_all_trajectories()
           (m_obs, m_ats, m_rs, m_ds, m_vs, m_logprobs) = tmp
           # copy data from one env run to training batch sample
           if obs is None:
               obs = m_obs
               ats = m_ats
               rs = m_rs
               ds = m_ds
               vs = m_vs
               logprobs = m_logprobs
           else:
               obs = np.concatenate((obs, m_obs), axis=0)
               ats = np.concatenate((ats, m_ats), axis=0)
               rs = np.concatenate((rs, m_rs), axis=0)
               ds = np.concatenate((ds, m_ds), axis=0)
               vs = np.concatenate((vs, m_vs), axis=0)
               logprobs = np.concatenate((logprobs, m_logprobs), axis=0)

        data = (obs, ats, rs, ds, vs, logprobs)
        # batch training, PPO, update weights
        self.update_weights(data)

        # deterministic test
        test_env = MhopEnv()
        test_env.set_map(self._map)
        observations, infos = test_env.reset()
        taken = []
        for i in range(gconf.TOTAL_HOPS - 1):
            observations = torch.from_numpy(observations).float()
            logits, values = self.model(observations)
            logits = torch.unsqueeze(logits, dim=0)
            # masking
            logits[self._map.M+1:] = -math.inf
            for t in taken:
                logits[0, t] = -math.inf
            logits[0, 0] = -math.inf
            actions = PolicyNN.sample_action(logits, deterministic=True)
            if i == 0:
                #print(torch.softmax(logits[0], dim=-1))
                #print(logits[0])
                pass
            taken.append(actions[0])
            # step
            observations_new, rewards, dones, truncates, infos = test_env.step(actions[0])
            observations = observations_new


        observations_new, rewards, dones, truncates, infos = test_env.step(0)
        #print(taken)


        # record
        self.eps += 1
        self.total_eps += 1

    def main_loop(self):
        for i in range(100000):
            self.one_epoch()

    def Init(self, load):
        """
        1. prepare envs
        2. load/create model
        3. init optimizer
        """
        # init envs: MP
        # seed
        # seed = 123

        if load == False:
            self.new_map()
        else:
            self.load_map()

        if gconf.EXPOSE_ALL:
            self.num_envs = self._map.M * (self._map.M - 1)
        else:
            self.num_envs = gconf.TOTAL_ENVS
        self.envs = gym.vector.make(
            "mhop-v0", num_envs=self.num_envs, asynchronous=False
        )
        self.envs.call("set_map", self._map)

        # init model
        # a. create model
        # b. load model
        self.OBS_SHAPE = self.envs.observation_space.shape[1]
        if load:
            self.model = torch.load("outputs/save_model")
        else:
            # all the maximum possible M with a terminate signal '0'
            model_output_shape = gconf.MAX_M + 1
            self.model = PolicyNN(self.OBS_SHAPE, model_output_shape)
        # if load_model is None:
        #    model.init_weights()
        # else:
        #    model.load_state_dict(torch.load(load_model))

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=gconf.lr)


def checkpoint(trainer):
    # save check point
    envs = trainer.envs
    model = trainer.model

    trainer.envs = None
    trainer.model = None

    torch.save(model, "outputs/save_model")

    with open("outputs/trainer", "wb") as f:
        pickle.dump(trainer, f)

    trainer.envs = envs
    trainer.model = model


def main():
    # load or create progress
    if os.path.exists("outputs/trainer"):
        # load
        with open("outputs/trainer", "rb") as f:
            trainer = pickle.load(f)
        print("===trainer loaded===")
        print(f"total epochs: {trainer.total_eps}")
        print(f"current epochs: {trainer.eps}")
        print(f"map_id: {trainer.map_id}")
        print(f"training best:")
        print(trainer.train_best)
        print("====================")
        trainer.Init(load=True)
    else:
        # create
        trainer = Trainer()
        trainer.Init(load=False)
        print("prog created")
        # checkpoint(trainer)

    trainer.main_loop()
    checkpoint(trainer)


if __name__ == "__main__":
    parms = sys.argv[1:]
    try:
        if parms[0] == "reset":
            os.remove("outputs/trainer")
            shutil.rmtree("outputs/maps")
            print("reset exp")
    except:
        pass

    main()
