import io
import pstats
import cProfile
import random

from partition import Partition, PartitionGenerator
from partition_util import partitionfun



# class BulgarianSolitairePath():
#     def __init__(self):
#         # List of strings
#         self.path = list()
    
#     def append(self, partition_string):
#         self.path.append(partition_string)

#     def contains(self, partition):


class BulgarianSolitaire():
    def __init__(self, start_position):
        self.position = start_position

        # List of partitions encountered during play, represented as strings
        self.path = list()

        # dict with position string as key and number of steps to cycle as value
        self.nsteps_to_cycle = dict()

    def play_until_first_recurrence(self):
        self.path.append(self.position.to_tuple())

        done = False
        while not done:
            self.position.bulgarian_solitaire_step()
            p_tuple = self.position.to_tuple()
            done = (p_tuple in self.path)
            if not done:
                self.path.append(p_tuple)

        # Create nsteps_to_cycle: When p_tuple found in
        # self.path, start incrementing number of steps to cycle
        cnt = 0
        found = False
        for i in reversed(range(len(self.path))):
            self.nsteps_to_cycle[self.path[i]] = cnt
            if not found:
                found = (self.path[i] == p_tuple)
            if found:
                cnt += 1


def mkhist(d):
    all_vals = list(d.values())
    out = list()
    for i in range(0, max(all_vals) + 1):
        out.append(all_vals.count(i))
    return out


# for i in range(1, 10):
#     n = int(i * (i + 1) / 2)
#     p = Partition([1] * n)
#     bs = BulgarianSolitaire(p)
#     bs.play_until_first_recurrence()
#     h = mkhist(bs.nsteps_to_cycle)

if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()

    for i in range(8, 17):
        n = int(i * (i + 1) / 2)
        n_partitions = partitionfun(n)
        pg = PartitionGenerator(n)
        n_steps_to_cycle = dict()
        partitions_done = list()
        while (pg.has_next()):
            p = pg.get_next()
            if random.random() < 0.01:
                r = round(len(partitions_done) / n_partitions * 1000000) / 10000
                print(f"{i} {n} ({n_partitions}): {r} %   ", end = "\r")
            p_tuple = p.to_tuple()
            if (p_tuple not in partitions_done):
                bs = BulgarianSolitaire(p)
                bs.play_until_first_recurrence()
                n_steps_to_cycle.update(bs.nsteps_to_cycle)
                for pdone in bs.nsteps_to_cycle:
                    if pdone not in partitions_done:
                        partitions_done.append(pdone)
       
        h = mkhist(n_steps_to_cycle)
        print(f"{i}: {h[0:11]}...")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
