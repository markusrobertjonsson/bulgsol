import math
from sympy import Symbol, series, Poly
from itertools import permutations
from partition import Partition
from partition_util import partitionfun

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
    '''
    partitions_out = []

    cycle_partitions_repr = [p.to_key() for p in cycle_partitions]
    for cycle_partition in cycle_partitions:
        rev_img, inds = cycle_partition.reversed_image()
        for p, ind in zip(rev_img, inds):
            if p.to_key() not in cycle_partitions_repr:
                partitions_out.append((cycle_partition, ind))

    return partitions_out


def level_stat_orbit(necklace):
    '''
    In the orbit with the cycle represented as the specified necklace, return
    the number of partitions with k steps to reach the cycle, for each k.
    '''
    out = list()

    # Level 0 - the cycle partitions
    cycle_partitions = get_partitions_from_necklace(necklace)
    out.append(len(cycle_partitions))

    # Level 1 - the first level out of cycle
    partitions = get_reversed_playable_out_of_cycle(cycle_partitions)
    curr_level = []
    for partition, part_ind in partitions:
        partition_rev = partition.copy()
        partition_rev.reversed_bulgarian_solitaire_step(part_ind)
        curr_level.append(partition_rev)
    out.append(len(curr_level))

    # Main loop - levels 2, 3, ...
    next_level = []
    done = False
    while not done:
        for partition in curr_level:
            rev_img, _ = partition.reversed_image()
            next_level.extend(rev_img)
        level_size = len(next_level)
        done = (level_size == 0)
        if not done:
            out.append(level_size)
            curr_level = next_level
            next_level = []

    return out


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


necklace_base = 'BW'
for k in range(1, 10):
    necklace = necklace_base * k
    # necklace1 = 'W' * k
    levels = level_stat_orbit(necklace)
    print(necklace)
    print(levels)

    # necklace2 = 'W' + 'BW' * k
    # levels2 = level_stat_orbit(necklace2)
    # print(levels2)
   
    print()

#A = [-5, -7, -12, -16, -9, 8, 1]
#B = [-1, 0, 0, 1, 4, 6]
A = [2, 1, -3]
B = [1, -1, -3, 1]
ind = 9
result = coeff_of_ratfun(A, B, ind, mult_by_1_minus_x=True)
print(f"The first {ind + 1} coefficients: {result}")


# for n in range(3, 50):
#     levels = level_stat(n)
#     sum1 = sum([sum(levels[k]) for k in levels])
#     n_partitions = partitionfun(n)
#     assert(n_partitions == sum1)
#     n_cycles = len(levels)
#     cycles_per_partitions = n_cycles / n_partitions
#     print(f"{n=}, {n_partitions=}, {n_cycles}, {cycles_per_partitions}")






import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

# Fixing random state for reproducibility
np.random.seed(19680801)


metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = 0, 0

with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()