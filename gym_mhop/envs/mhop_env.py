import sys

sys.path.append("../../")
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .Maps import Map
from .Maps import MapGenParms
from GlobalConfig import GlobalConfig as gconf
import copy

from gym.spaces.box import Box
from gym.spaces import Discrete


class MhopEnv(gym.Env):
    r"""
    Basic multihop relaying class. Choosing relays hop by hop.
    """

    def __init__(self, seed=None):
        """
        Initialize the environment via:
        1. generation (random)
        2. generation (by seed)
        3. loading local file
        """
        self.route_mx = None
        self.route_ls = None

        # ======== define spaces =========
        # TODO: make an adaptive observation size
        #self.observation_space = Box(low=-1.0, high=1.0, shape=(92,), dtype=np.float64)
        # self.observation_space = Box(low=-1.0, high=1.0, shape=(277,), dtype=np.float64)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(5357,), dtype=np.float64)
        # self.observation_space = Box(low=-1.0, high=1.0, shape=(1432,), dtype=np.float64)
        # self.observation_space = Box(low=-1.0, high=1.0, shape=(20707,), dtype=np.float64)
        self.action_space = Discrete(gconf.MAX_M)

    @staticmethod
    def generate_map():
        _map = Map(gen_parms=gconf.gen_parms)
        _map.generate()
        return _map

    def set_map(self, m=None):
        if m is None:
            # generate
            self._map = Map(gen_parms=gconf.gen_parms)
            self._map.generate()
        else:
            self._map = m

        self.sender_node = self._map.furthest_pair_ls[0]
        self.receiver_node = self._map.furthest_pair_ls[1]

        # ** always padding -> flatten
        M = self._map.M
        padded_G = np.zeros((gconf.MAX_M, gconf.MAX_M, gconf.gen_parms.K))
        padded_G[0:M, 0:M, :] = self._map.G[:, :, :]
        self.flatten_G = padded_G.flatten()

        # get half G
        size_half = int(gconf.MAX_M * (gconf.MAX_M - 1) / 2)
        self.half_G = np.zeros(size_half)
        count = 0
        for i in range(M):
            for j in range(i):
                self.half_G[count] = padded_G[i, j]
                count += 1

        # preprocessing: clip
        clip = gconf.G_CLIP
        self.half_G[np.where(self.half_G > clip)] = clip

    def step(self, action: int):
        """
        Action Space: Discrete(M)
        Gives warning when the action is not feasible:
        1. No such relay
        2. Relay has been selected before
        """

        # if self.n_selected == gconf.TOTAL_HOPS - 1:
        #    #reward = 1 - self.overall_performance
        #    reward = self.overall_performance
        # else:
        #    reward = 0
        info = {}

        if action == 0:
            self.update_performance()
            reward = self.overall_performance * 10**3
            done = True
            info["route"] = self.route_ls
        else:
            # take action
            self.route_ls.append(action)
            self.route_mx[self.n_selected, int(action)] = 1
            self.n_selected += 1

            reward = 0
            done = False

        # if len(self.route_ls) == gconf.TOTAL_HOPS:
        #    done = True
        # else:
        #    done = False

        truncated = False

        observation = self.make_observation(action=action)

        return observation, reward, done, truncated, info
        # return observation, reward, done, info

    def update_performance(self):
        # rewards only given at the last selection
        # require normalization as all rewards are positive
        tmp = copy.deepcopy(self.route_ls)
        tmp.append(self.receiver_node)

        # all_sub: vector of all subcarriers
        # overall: a scalar
        self.all_sub_performance = self._map.route_performance(tmp, verbose=False)
        self.overall_performance = np.min(self.all_sub_performance)

    def get_nodes_to_mask(self):
        return self.route_ls + [self.receiver_node]

    def make_observation(self, action: int = -1):
        """
        Observation:
        Space: TODO: ?
        1. Global G
        2. chosen nodes
        3. picking node
        4. overall performance
        5. number of remaining relays to choose
        """

        remaining_steps = (gconf.TOTAL_HOPS - len(self.route_ls)) / 10

        con_args = (
            np.array(self.route_mx.flatten()),
            np.array([action / 100]),
            # np.array(current_performance),
            np.array([self.overall_performance]),
            np.array([remaining_steps]),
            self.half_G,
        )
        observation = np.concatenate(con_args)

        #print(f"observation dimension: {len(observation)}")
        return observation

    def reset(self, seed=None, options=None):
        """
        1. Clear the selections.
        2. Return the first frame.
        """
        self.route_mx = np.zeros((gconf.TOTAL_HOPS + 1, gconf.MAX_M + 1))
        self.route_mx[0, self.sender_node] = 1
        self.route_mx[gconf.TOTAL_HOPS, self.receiver_node] = 1
        self.n_selected = 0
        self.route_ls = []
        self.route_ls.append(self.sender_node)

        self.overall_performance = 0
        self.all_sub_performance = np.zeros(gconf.gen_parms.K)

        observation = self.make_observation()
        info = {}
        info["route"] = None
        return observation, info

    def render(self):
        # TODO: render process to Unity
        pass

    def close(self):
        """
        Cleanup.
        """
        # print("env: close not implemented")
        pass
