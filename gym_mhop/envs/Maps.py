import numpy as np
import itertools

import requests
import torch

import math


class MapGenParms:
    """
    The parameter set to generate a map.
    """

    def __init__(self):
        self.K = 1
        self.MIN_M = 8
        self.MAX_M = 10
        self.alpha = 2.0
        self.Pt = 1.0
        self.N0 = 1.0
        self.lambda_p = 3.0
        self.lambda_o = 9.0

        self.lambda_p = 5.0
        self.lambda_o = 7.0
        self.radius = 200  # radius of parent disk
        self.rect = (1000, 1000)

        self.gen_method = 1

    def __str__(self):
        s = f"K={self.K},alpha={self.alpha},Pt={self.Pt},"
        s += f"N0={self.N0},lambda_o={self.lambda_o},"
        s += f"lambda_p={self.lambda_p},radius={self.radius},"
        s += f"rect={self.rect}"
        return s


class Map:
    """
    Env version, keep core functions only
    """

    def __init__(self, gen_parms=None):
        # spatial position (x, y)
        # M * 2
        self.M = None
        self.pos = None

        # if the node is a parent node, 0/1
        # M * 1
        self.is_parent = None

        # distance tensor
        # M * M * 1
        self.dis = None

        # global g
        # M * M * K
        self.g = None

        # global CSI
        # M * M * K
        self.G: np.ndarray = None

        self.gen_parms = gen_parms
        self.num_parents = 0
        self.map_id = -1

    def generate(self):
        parms = self.gen_parms
        parents, offsprings = Map.gen_matern_pos(
            self.gen_parms, min_n=parms.MIN_M, max_n=parms.MAX_M
        )
        self.pos = np.concatenate((parents, offsprings), axis=0)
        self.M = len(self.pos)
        self.is_parent = np.full(self.M, False)
        self.is_parent[0 : len(parents)] = True

        self.gen_SNR()

    def route_performance(self, route: list, verbose=False):
        """
        Input: route:List, example [5, 8, 3]
        Return: performance: float
                0 if not valid
        """
        # for calculation only
        route[:] = [x - 1 for x in route]
        try:
            mx = np.full((len(route)-1, self.gen_parms.K), fill_value=math.inf)
            for i in range(len(route) - 1):
                begin_node = route[i]
                next_node = route[i + 1]
                mx[i,:] = self.G[begin_node, next_node, :]
                if verbose:
                    tmp_min = np.min(mx, axis=0)
                    print(f"{begin_node+1}->{next_node+1}: {self.G[begin_node, next_node, :]} => {tmp_min}")
            return np.min(mx, axis=0)
        except Exception as ex:
            if verbose:
                print(f'Invalid route!: {route}')
            return np.zeros(self.gen_parms.K)

    def gen_SNR(self):
        """
        Generate SNR based on Matern Pos
        """
        parms = self.gen_parms
        K = parms.K
        M = self.M

        self.dis = np.zeros((M, M))
        pairs = list(itertools.permutations(range(M), 2))
        for p in pairs:
            self.dis[p[0], p[1]] = np.linalg.norm(self.pos[p[0]] - self.pos[p[1]])

        # get furthest pair
        self.furthest_distance = np.amax(self.dis)
        pair = np.where(self.dis == self.furthest_distance)[0]
        pair += 1
        self.furthest_pair = f"{pair[0]}-{pair[1]}"
        self.furthest_pair_ls = pair

        # global g
        self.g = np.random.exponential(scale=1, size=(M, M, K))
        for i in range(M):
            for j in range(i + 1, M):
                self.g[j][i] = self.g[i][j]
            # self to 0
            self.g[i][i] = np.zeros((K))

        # global CSI
        pow_mx = np.power(self.dis, -parms.alpha, where=self.dis > 0)
        self.G = self.g**2 * np.repeat(pow_mx[:, :, np.newaxis], K, axis=2)
        self.G[np.isinf(self.G)] = 0.0
        self.G[np.isnan(self.G)] = 0.0

    @staticmethod
    def gen_matern_pos(parms, parents_=None, offsprings_=None, min_n=5, max_n=50):
        # Copy related parameters
        radius = parms.radius
        rect = parms.rect
        lambda_o = parms.lambda_o
        lambda_p = parms.lambda_p

        # area -> km^2
        total_area = rect[0] * rect[1] / 10**6

        # generate parent nodes
        # Poisson point process
        while True:
            if parents_ is not None:
                num_parents = parents_
                num_offsprings = offsprings_
                break
            else:
                num_parents = np.random.poisson(total_area * lambda_p)
                num_offsprings = np.random.poisson(lambda_o, num_parents)

            total_n = np.sum(num_offsprings) + num_parents

            if total_n >= min_n and total_n <= max_n:
                break

        # num_parents = 50

        # generate parent positions
        x_parents = rect[0] * np.random.uniform(0, 1, num_parents)
        y_parents = rect[1] * np.random.uniform(0, 1, num_parents)

        total_offsprings = np.sum(num_offsprings)

        # generate offsprings locations in polar coordinates
        theta = 2 * np.pi * np.random.uniform(0, 1, total_offsprings)
        rho = radius * np.random.uniform(0, 1, total_offsprings)

        # Convert from polar to Cartesian coordinates
        xx0 = rho * np.cos(theta)
        yy0 = rho * np.sin(theta)

        # replicate parent points (ie centres of disks/clusters)
        xx = np.repeat(x_parents, num_offsprings)
        yy = np.repeat(y_parents, num_offsprings)

        # translate points (ie parents points are the centres of cluster disks)
        xx = xx + xx0
        yy = yy + yy0

        # thin points if outside the simulation window
        booleInside = (xx >= 0) & (xx <= rect[0]) & (yy >= 0) & (yy <= rect[1])
        # retain points inside simulation window
        xx = xx[booleInside]
        yy = yy[booleInside]

        parents = np.concatenate(
            (x_parents.reshape(-1, 1), y_parents.reshape(-1, 1)), axis=1
        )
        offsprings = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

        return parents, offsprings
