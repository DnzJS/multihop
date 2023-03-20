import sys

from numba import cuda
import math
import numpy as np
import requests
#from multihop import ExpConfig
#from multihop import Map
from gym_mhop.envs.Maps import Map
from gym_mhop.envs.Maps import MapGenParms
import itertools

from datetime import datetime


@cuda.jit
def cuda_solve(
    G_mx,
    result_mx,
    result_final,
    factoradics,
    flags,
    permus,
    total_permutations,
    thread_runs,
    pos,
    total_nodes,
    total_subcarriers,
    relays,
    begin_node,
    end_node,
):
    # get idx
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    total_threads = cuda.blockDim.x * cuda.gridDim.x
    # print(total_threads, cuda.blockDim.x, cuda.gridDim.x)

    run = 0
    # decimal_code = pos + run * total_threads + idx

    for run in range(thread_runs):
        decimal_code = pos + run * total_threads + idx
        if decimal_code >= total_permutations:
            return

        # decimal -> factoradic
        objects = total_nodes
        samples = relays
        n = decimal_code
        for i in range(samples):
            x = 1
            for j in range(objects - samples + 1, objects - i):
                x *= j
            digit = math.floor(n / x)
            factoradics[run * total_threads + idx, i] = digit
            n = n % x

        # factoradic -> permutation
        for i in range(total_nodes):
            flags[idx, i] = False
        for i in range(relays):
            c = 0
            pointer = 0

            while True:
                if c == factoradics[idx, i] and flags[idx, pointer] == False:
                    break
                if not flags[idx, pointer]:
                    c += 1
                pointer += 1

            permus[idx, i] = pointer
            flags[idx, pointer] = True

        # evaluate permutation
        for k in range(total_subcarriers):
            # TODO: start and end nodes
            # nodes list: [begin, relay1, relay2, end]
            # idx in that batch
            idx_batch = run * total_threads + idx
            # init with the first hop
            result_mx[idx_batch, k] = G_mx[begin_node, permus[idx_batch, 0], k]
            # relays
            for i in range(relays):
                cuda.atomic.min(
                    result_mx,
                    (idx_batch, k),
                    G_mx[permus[idx_batch, i], permus[idx_batch, i + 1], k],
                )
            # calculate the last hop
            cuda.atomic.min(
                result_mx,
                (idx_batch, k),
                G_mx[permus[idx_batch, relays - 1], end_node, k],
            )

            # result_mx[run*total_threads + idx, k] *= 10**7
            # result_mx[run*total_threads + idx, k] = decimal_code

        # record result
        # result_final[run*total_threads + idx] = decimal_code
        result_final[idx_batch] = result_mx[idx_batch, 0]
        for k in range(1, total_subcarriers):
            cuda.atomic.min(result_final, idx_batch, result_mx[idx_batch, k])


def save_tmp_exp_rec(map_id, begin, end, hops, route, value):
    SERVER_URL = "http://127.0.0.1:10000"

    parms = {
        "map_id": map_id,
        "source": begin,
        "end": end,
        "hops": hops,
        "route": route,
        "value": value,
    }
    url = SERVER_URL + "/unity_tmp_exp_rec"
    r = requests.get(url, params=parms)


def save_run_rec(parms):
    SERVER_URL = "http://127.0.0.1:10000"

    url = SERVER_URL + "/save_run_rec"
    r = requests.get(url, params=parms)


class CUDA_BF_Solver:
    def __init__(self):
        pass

    def load(self, map_id):
        SERVER_URL = "http://127.0.0.1:10000"

        # load map
        self.map_id = map_id
        parms = {"map_id": map_id}
        url = SERVER_URL + "/unity_get_map"
        r = requests.get(url, params=parms)
        self.config, self.map = self.make_map(str(r.content))

    def assign_map(self, config, _map):
        self.config = config
        self.map = _map

        if type(self.map.furthest_pair) == str:
            tmp = self.map.furthest_pair.split("-")
            self.map.furthest_pair = [int(x) for x in tmp]

    def solve(self, hops, begin_node_=None, end_node_=None, use_gae=False, ref=None):
        t0 = datetime.now()
        larger = 0  # counts the number of solutions that has better performance than ref
        if begin_node_ == None:
            print(self.map.furthest_pair)
            begin_node = self.map.furthest_pair[0]
            end_node = self.map.furthest_pair[1]
        else:
            begin_node = begin_node_
            end_node = end_node_

        # RTX3070 CUDA cores: 5888
        blocks_per_grid = 20
        threads_per_block = 250
        # How many times a CUDA thread calculates in a batch
        calculations_per_batch = 100

        # blocks_per_grid = 2
        # threads_per_block = 10
        # calculations_per_batch = 10

        batch_size = threads_per_block * blocks_per_grid * calculations_per_batch

        # P(n,r): n - objects, r - samples, get total permutation number
        # P(n,r) = n!/(n-r)!
        n = self.map.M
        # relays = hops - 1
        r = relays = hops - 1
        total_permutations = 1
        for x in range(n - r + 1, n + 1):
            total_permutations *= x

        #print("******")
        #print(f"total_permutations={total_permutations}")
        #print(f"batch_size: {batch_size}")

        cuda_pos = 0
        global_best_permu = []
        global_best = -1

        flag_device = cuda.device_array((batch_size, self.map.M), dtype=bool)
        factoradic_device = cuda.device_array((batch_size, relays), dtype=int)
        permu_device = cuda.device_array((batch_size, relays), dtype=int)
        result_mx = cuda.device_array((batch_size, self.config.K))
        result_final_mx = cuda.device_array((batch_size))

        percentage_step = 0.0001
        percentage_step = 0.05
        percentage_print = percentage_step
        batch_count = 0
        while cuda_pos < total_permutations:

            percentage = cuda_pos / total_permutations
            if percentage > percentage_print:
                t1 = datetime.now()
                #print(
                #    f"{cuda_pos} / {total_permutations}, {percentage * 100}%, "
                #    f"avg_batch_time: {(t1-t0).total_seconds()/batch_count}, "
                #    f"calculations per second: {cuda_pos/(t1-t0).total_seconds()}"
                #)
                percentage_print += percentage_step

            batch_count += 1
            # G_mx_device = cuda.to_device(self.map.G)
            if use_gae:
                G_mx_device = cuda.to_device(self.map.G_ae)
            else:
                G_mx_device = cuda.to_device(self.map.G * 10**7)

            cuda_solve[blocks_per_grid, threads_per_block](
                G_mx_device,
                result_mx,
                result_final_mx,
                factoradic_device,
                flag_device,
                permu_device,
                total_permutations,
                calculations_per_batch,
                cuda_pos,
                self.map.M,
                self.config.K,
                relays,
                begin_node,
                end_node,
            )

            # copy result from devices to host
            result_host = result_mx.copy_to_host()

            if ref is not None:
                index = np.where(result_host == ref*10**4)
                larger += (result_host > ref).sum()

            factoradic_host = factoradic_device.copy_to_host()
            permus_host = permu_device.copy_to_host()
            result_final_host = result_final_mx.copy_to_host()
            # compare and record the best results
            batch_best = np.amax(result_final_host)
            batch_best_permu = np.resize(
                permus_host[np.argmax(result_final_host)], relays
            )
            batch_best_permu = np.insert(batch_best_permu, 0, begin_node)
            batch_best_permu = np.insert(
                batch_best_permu, len(batch_best_permu), end_node
            )
            # print(f"cuda_pos: {cuda_pos}")
            # print("batch: ", batch_best, batch_best_permu)
            if cuda_pos == 0 or batch_best > global_best:
                global_best = batch_best
                global_best_permu = batch_best_permu
                print(f"global best: {global_best} {global_best_permu+1}")
            # next batch
            cuda_pos += batch_size

        print("size", (result_host.nbytes / 1024 / 1024), "mb")
        print(f"total run time: {datetime.now()-t0}")
        print("done")

        return global_best_permu, global_best, larger, total_permutations
        # record result
        route_str = "-".join([str(element) for element in global_best_permu])
        # save_tmp_exp_rec(self.map_id, begin_node, end_node, hops, route_str, global_best)
        parms = {
            "map_id": map_id,
            "source": begin_node,
            "destination": end_node,
            "approach": "'BF'",
            "num_hops": hops,
            "route": f"'{route_str}'",
            "fitness": global_best,
        }
        # save_run_rec(parms)

    def make_map(self, msg):
        lines = msg.split("<br>")
        exp_info_raw = lines[0].replace("b'", "").split(",")

        # M, K, alpha, Pt, N0
        config = ExpConfig()
        # Map
        map = Map()

        map.M = M = int(exp_info_raw[0])
        map.furthest_distance = float(exp_info_raw[5])
        pair_raw = exp_info_raw[6].split("-")
        map.furthest_pair = [int(pair_raw[0]), int(pair_raw[1])]
        config.K = K = int(exp_info_raw[1])
        config.alpha = float(exp_info_raw[2])
        config.Pt = float(exp_info_raw[3])
        config.N0 = float(exp_info_raw[4])

        # pos, is_parent
        map.pos = np.empty([M, 2])
        map.is_parent = [None] * M
        for m in range(M):
            temp = lines[2 + m].split(",")
            map.pos[m] = [float(temp[0]), float(temp[1])]
            map.is_parent[m] = temp[2] == "1"

        map.g = np.empty([M, M, K])
        map.G = np.empty([M, M, K])
        for i in range(M):
            for j in range(M):
                # why 3: there are two *** and one config line
                g_i_to_j = lines[3 + M + i * M + j].split(",")
                G_i_to_j = lines[(4 + M + M * M) + i * M + j].split(",")
                for k in range(K):
                    map.g[i, j, k] = float(g_i_to_j[k])
                    map.G[i, j, k] = float(G_i_to_j[k])

        return config, map




if __name__ == "__main__":
    pass
    #maps = range(104, 141)
    #for map_id in maps:
    #    print(f"map_id: {map_id}")
    #    solver = CUDA_BF_Solver()
    #    solver.load(map_id)
    #    solver.solve(hops=3)
    #    solver.solve(hops=4)
    #    # solver.solve(hops=5)

    #sys.exit()

    #perm = itertools.permutations([1, 2, 3, 4, 5, 6], 3)
    #c = 0
    #for p in perm:
    #    print(f"{c}: {p}")
    #    c += 1

    #objects = 6
    #samples = 3
    #n = 80
    #r = []
    #for i in range(samples):
    #    x = 1
    #    for j in range(objects - samples + 1, objects - i):
    #        x *= j
    #    digit = math.floor(n / x)
    #    r.append(digit)
    #    n = n % x

    #print(r)
