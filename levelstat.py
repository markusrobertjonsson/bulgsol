import math
from sympy import Symbol, series, Poly
from itertools import permutations
from partition import Partition
from partition_util import partitionfun
from scipy.sparse import lil_matrix

from treeplot import treeplot
import matplotlib.pyplot as plt


def triangular_number(ind):
    return ind * (ind + 1) // 2


def largest_tri_smaller_than(n):
    '''
    Return the largest triangular number smaller than or equal to n, and the
    corresponding index in the sequence of triangular numbers.
    '''
    if n <= 0:
        return 0, 0  # There are no positive triangular numbers less than or equal to 0
    ind = int((math.sqrt(1 + 8 * n) - 1) // 2)
    return  triangular_number(ind), ind


def rotate(seq, n):
    return seq[n:] + seq[:n]


def is_canonical_necklace(necklace):
    n = len(necklace)
    for i in range(1, n):
        if rotate(necklace, i) < necklace:
            return False
    return True


def coeff_of_ratfun(A, B, ind, mult_by_1_minus_x=False):
    '''
    Returns the coefficients up to and including ind in the series expansion of
    the rational function A/B.

    Example usage:
    A = [1, -1]  # Represents 1 - x
    B = [1, -3, 1]  # Represents 1 - 3x + x^2
    ind = 3
    result = coeff_of_genfun(A, B, i)
    print(f"The first {i + 2} coefficients of the series expansion of A/B is: {result}")
    '''
    x = Symbol('x')

    # Create polynomials from coefficient lists
    poly_A = Poly(A[::-1], x)
    poly_B = Poly(B[::-1], x)

    # Compute the series expansion of A / B up to x^i
    if mult_by_1_minus_x:
        C = [1, -1]  # 1-x
        poly_C = Poly(C[::-1], x)
        ratfun = poly_C.as_expr() * poly_A.as_expr() / poly_B.as_expr()
    else:
        ratfun = poly_A.as_expr() / poly_B.as_expr()

    series_expansion = ratfun.series(x, 0, ind+1)

    # Extract the coefficients up to ind
    out = []
    for i in range(ind + 1):
        coeff = series_expansion.coeff(x, i)
        out.append(coeff)

    return out


def generate_necklaces(n_black, n_white):
    '''Return all necklaces with n_black black beads and n_white white beads.'''

    # Generate all unique permutations
    beads = 'B' * n_black + 'W' * n_white
    unique_permutations = set(permutations(beads))

    # Filter out non-canonical necklaces
    canonical_necklaces = set()
    for perm in unique_permutations:
        if is_canonical_necklace(perm):
            perm_str = ''.join(perm)
            canonical_necklaces.add(perm_str)

    return canonical_necklaces


def get_partition_from_necklace_repr(necklace):
    necklace_len = len(necklace)
    parts = []
    for i in reversed(range(necklace_len - 1)):
        part = i + 1
        if necklace[i] == 'B':
            part += 1
        parts.append(part)
    if necklace[-1] == 'B':
        parts.append(1)
    return Partition(parts)


def get_partitions_from_necklace(necklace):
    rotated_necklaces = [necklace]

    partition = get_partition_from_necklace_repr(necklace)
    partitions = [partition]

    necklace_len = len(necklace)
    for i in range(necklace_len - 1):
        rotated_necklace = rotate(necklace, i + 1)
        if rotated_necklace not in rotated_necklaces:
            rotated_necklaces.append(rotated_necklace)
            partition = get_partition_from_necklace_repr(rotated_necklace)
            partitions.append(partition)
    return partitions


def get_reversed_playable_out_of_cycle(cycle_partitions):
    '''
    Return the partitions in the specified list that leads out of the
    cycle when played backwards.

    @return List of tuples. Each tuple is
    (partition_in_cycle, playable_row, index_to_cycle_partitions)
    '''
    partitions_out = []

    cycle_partitions_repr = [p.to_key() for p in cycle_partitions]
    for cycle_partition in cycle_partitions:
        rev_img, inds = cycle_partition.reversed_image()
        for p, ind in zip(rev_img, inds):
            if p.to_key() not in cycle_partitions_repr:
                partitions_out.append((cycle_partition, ind))

    return partitions_out


class AdjMatrix():
    def __init__(self):
        # A list of lists, first index is row, second column
        self.matrix = list()
    
    def add(self, row, to_append):
        curr_len = len(self.matrix)
        if row >= curr_len:
            for _ in range(curr_len, row + 1):
                self.matrix.append(list())
        self.matrix[row].append(to_append)

    def max(self):
        matrix_max = -1
        for x in self.matrix:
            for xx in x:
                matrix_max = max(xx, matrix_max)
        return matrix_max

    def to_sparse(self):
        matrix_max = self.max()
        curr_len = len(self.matrix)
        if curr_len < matrix_max:
            for _ in range(curr_len, matrix_max):
                self.matrix.append(list())

        n = len(self.matrix)
        assert(matrix_max == n), f"{matrix_max=}, {n=}"

        sparse = lil_matrix((n + 1, n + 1))
        # sparse = [[0] * (n + 1) for _ in range(n + 1)]
        for row in range(len(self.matrix)):
            cols = self.matrix[row]
            for col in cols:
                # sparse[row][col] = 1
                sparse[row, col] = 1
        return sparse


def level_stat_orbit(necklace, depth=None):
    '''
    In the orbit with the cycle represented as the specified necklace, return
    the number of partitions with k steps to reach the cycle, for each k=0,...,depth.
    If depth is None or not specified, the entire orbit is returned.
    '''
    level_sizes = list()
    adj_matrix = AdjMatrix()

    node = 0

    # Level 0 - the cycle partitions
    cycle_partitions = get_partitions_from_necklace(necklace)
    level_sizes.append(len(cycle_partitions))

    # Level 1 - the first level out of cycle
    partitions = get_reversed_playable_out_of_cycle(cycle_partitions)
    curr_level = []
    for partition, part_ind in partitions:
        partition_rev = partition.copy()
        partition_rev.reversed_bulgarian_solitaire_step(part_ind)
        node += 1
        curr_level.append((partition_rev, node))
        adj_matrix.add(0, node)
    level_sizes.append(len(curr_level))

    # Main loop - levels 2, 3, ...
    level_cnt = 2
    next_level = []
    done = False
    while not done:
        for partition, node_ind in curr_level:
            rev_img, _ = partition.reversed_image()
            
            # next_level.extend(rev_img)
            for p in rev_img:
                node += 1
                next_level.append((p, node))
                adj_matrix.add(node_ind, node)

        level_size = len(next_level)
        done = (level_size == 0)
        if not done:
            level_sizes.append(level_size)
            curr_level = next_level
            next_level = []
            level_cnt += 1
            done = (level_cnt == depth)  # Stop if we reached the specified depth

    m = max([max(x) for x in adj_matrix.matrix if len(x) > 0])
    return level_sizes, adj_matrix.to_sparse()


def level_stat(n):
    '''
    For Bulgarian solitaire with n cards, find the cycles and return the number
    of partitions with k steps to reach the cycle, for each k, for each cycle.
    '''
    tri, n_parts = largest_tri_smaller_than(n)
    n_black = n - tri
    n_white = n_parts + 1 - n_black
    necklaces = generate_necklaces(n_black, n_white)

    levels = dict()

    for necklace in necklaces:
        levels_orbit = level_stat_orbit(necklace)
        levels[necklace] = levels_orbit

    return levels

def levelstat_plot(levels, ax, bar_max=None):
    x = [str(i) for i in range(len(levels))]
    ax.bar(x, levels)  #, label=bar_labels)  # , color=bar_colors)
    ax.set_xlabel('number of partitions')
    ax.set_ylabel('distance to cycle')
    if bar_max is not None:
        ax.set_ylim([0, bar_max + 1])

# Test of treeplot

# necklace_base = 'BBWW'
# necklace_base = 'W'

# k_start = 4
# k_end = 8

# n_subplots = k_end - k_start + 1
# fig, axes = plt.subplots(1, n_subplots, figsize=(15, 8))
# axi = 0
# for k in range(k_start, k_end + 1):
#     necklace = necklace_base * k
#     # necklace1 = 'W' * k
#     levels, adj_matrix = level_stat_orbit(necklace, depth=5)
#     print(necklace)
#     print(levels)
#     treeplot(adj_matrix, ax=axes[axi])
#     axi += 1
#     print()
# plt.show()

k = 3
necklace = "BWWBWWW" * k
depth = 30
levels, _ = level_stat_orbit(necklace, depth=depth)
print(levels)

A = [-7, -10, -18, -33, -53, -54, -12, 23, 2]
B = [-1, 0, 0, 0, 1, 6, 14, 12]
result = coeff_of_ratfun(A, B, depth - 1, mult_by_1_minus_x=True)
print(f"The first {depth + 1} coefficients:\n{result}")


# Test of coeff_of_ratfun

# A = [2, 1, -3]
# B = [1, -1, -3, 1]
# ind = 9
# result = coeff_of_ratfun(A, B, ind, mult_by_1_minus_x=True)
# print(f"The first {ind + 1} coefficients: {result}")
