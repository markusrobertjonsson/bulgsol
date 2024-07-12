import io
import pstats
import cProfile
import random
import time

from partition import accel_asc
from partition_util import partitionfun

# class BulgarianSolitairePath():
#     def __init__(self):
#         # List of strings
#         self.path = list()
    
#     def append(self, partition_string):
#         self.path.append(partition_string)

#     def contains(self, partition):


class BulgarianSolitaire():
    # Types of data to collect
    # LEVEL_SIZE = 0
    # CYCLE = 0

    def __init__(self):
        pass
        # self.data_to_collect = []

        # Dict with position string as key and number of steps to cycle as value
        self.level_sizes = dict()

    # def add_data_to_collect(self, data):
    #     self.data_to_collect.append(data)

    def play_until_first_recurrence(self, start_position, start_position_key):
        position = start_position
        p_key = start_position_key

        if p_key in self.level_sizes:  # Already been here in the game graph
            return self.level_sizes[p_key]

        # List of partitions encountered during play, represented as tuples
        path = [p_key]

        done = False
        while True:  # not done:
            position.bulgarian_solitaire_step()
            p_key = position.to_key()

            done = (p_key in self.level_sizes)  # Already been here in the game graph
            if done:
                return self.generate_level_sizes_from_already_visited(path, p_key)

            self.level_sizes[p_key] = 0  # this will be set to non-0 in generate_level_sizes_from_already_visited

            done = (p_key in path)  # Found a recurring position
            if not done:
                path.append(p_key)
            else:
                return self.compute_nsteps_to_cycle(path, p_key)

    def has_played(self, p_key):
        return p_key in self.level_sizes

    def generate_level_sizes_from_already_visited(self, path, p_key):
        cnt = self.level_sizes[p_key]
        for i in reversed(range(len(path))):
            cnt += 1
            self.level_sizes[path[i]] = cnt
        return cnt


    def compute_nsteps_to_cycle(self, path, first_recurrence):
        # When first_recurrence found in path, start incrementing number of steps to cycle
        cnt = 0
        found = False
        for p_key in reversed(path):
            self.level_sizes[p_key] = cnt
            if not found:
                found = (p_key == first_recurrence)
            if found:
                cnt += 1
        return cnt - 1

def mkhist(d):
    all_vals = list(d.values())
    maxval = max(all_vals)
    out = [0] * (maxval + 1)
    for val in all_vals:
        out[val] += 1
    return out


# for i in range(1, 10):
#     n = int(i * (i + 1) / 2)
#     p = Partition([1] * n)
#     bs = BulgarianSolitaire(p)
#     bs.play_until_first_recurrence()
#     h = mkhist(bs.level_size)


if __name__ == "__main__":

    # # Speed test of accel_gen
    # for n in range(10, 11):
    #     tic = time.time()
    #     gen = accel_asc(n)
    #     for p in gen:
    #         print(p)
    #     toc = time.time()
    #     print(f"n = {n}: Elapsed time = {toc - tic}")

    # print()

    # # Speed test of PartitionGenerator
    # for n in range(10, 11):
    #     tic = time.time()
    #     gen = PartitionGenerator(n)
    #     while (gen.has_next()):
    #         p = gen.get_next()
    #         print(p)
    #     toc = time.time()
    #     print(f"n = {n}: Elapsed time = {toc - tic}")


    pr = cProfile.Profile()
    pr.enable()
    tic = time.time()

    for k in [10]:  # 10, 11, 12, 13, 14, 15, 16]:
        n = int(k * (k + 1) / 2)
        n_partitions = partitionfun(n)
        pg = accel_asc(n)
        bs = BulgarianSolitaire()
        # h = [0] * (2 + k * (k - 1))  # Max number of steps
        cnt = 0
        for p in pg:
            # print(f"p = {p.nparts}")
            if bs.has_played(p):
                continue
            p_key = p.to_key()
            cnt += 1
            if cnt % 10000 == 0:  # random.random() < 0.001:
                r = round(cnt / n_partitions * 1000000) / 10000
                print(f"{k} {n} ({n_partitions}): {r} %   ", end = "\r")
            n_steps = bs.play_until_first_recurrence(p, p_key)
            # print(f" {n_steps} moves")
            # h[n_steps] += 1

        # print(f"{k}: {h[0: 11]}...")
        h = mkhist(bs.level_sizes)
        print(f"{k}: {h[0:11]}...")

    toc = time.time()
    print(f"This took {(toc - tic)}")
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
