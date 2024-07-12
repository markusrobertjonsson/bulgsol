import bisect
import time
import hashlib


def get_first_occurrence_indices(sorted_list):
    if not sorted_list:
        return []
    unique_indices = [0]  # The first element is always a unique occurrence in a sorted list
    for i in range(1, len(sorted_list)):
        if sorted_list[i] != sorted_list[i - 1]:
            unique_indices.append(i)
    return unique_indices

# Example usage:
#sorted_list = [1, 2, 2, 3, 4, 4, 5]
#unique_indices = get_first_occurrence_indices(sorted_list)
#print(unique_indices)  # Output: [0, 1, 3, 4, 6]


class Partition():
    def __init__(self, parts, do_sort=True):
        # The integer of which this is a partition
        self.n = sum(parts)

        # The list of parts
        self.parts = parts

        if do_sort:
            self.sort_and_remove_nonpos_parts()

    def copy(self):
        return Partition(list(self.parts), do_sort=False)

    def __repr__(self):
        return str(self.parts)

    def __str__(self):
        return str(self.parts)

    def ind1(self, ind):
        assert(ind >= 1), "Index must be 1, 2, 3, ..."
        if ind - 1 < len(self.parts):
            return self.parts[ind - 1]
        else:
            return 0

    def get_nparts(self):
        return len(self.parts)

    def sort_and_remove_nonpos_parts(self):
        if self.get_nparts() == 0:
            return
        self.parts.sort(reverse=True)
        done = False
        while not done:
            last_ind = self.get_nparts() - 1
            done = (self.parts[last_ind] > 0)
            if not done:
                self.parts.pop()

    def to_key(self):
        # return str(self.parts)
        return tuple(self.parts)
        # return hash(tuple(self.parts)) // 10000000
        # hash = hashlib.sha1("my message".encode("UTF-8")).hexdigest()
        # return hash[:10]
        # return self._key()


    def bulgarian_solitaire_step(self):
        n_piles = self.get_nparts()
        for i in range(n_piles):
            self.parts[i] -= 1
            if self.parts[i] == 0:
                # self.parts = self.parts[(i + 1):]
                self.parts = self.parts[0: i]
                break
        # bisect.insort_left(self.parts, n_piles)
        rev = list(reversed(self.parts))
        bisect.insort_left(rev, n_piles)
        self.parts = list(reversed(rev))

    def reversed_is_playable(self, ind):
        return (self.parts[ind] >= len(self.parts) - 1)

    def reversed_bulgarian_solitaire_step(self, ind):
        assert(self.reversed_is_playable(ind)), f"Part {ind} is not playable"
        nparts = len(self.parts)
        part = self.parts.pop(ind)
        for i in range(nparts - 1):
            self.parts[i] += 1
        for i in range(part - nparts + 1):
            self.parts.append(1)

    def reversed_playable_inds(self):
        playable_inds = []
        curr = None
        for ind in range(self.get_nparts()):
            if self.parts[ind] != curr:
                curr = self.parts[ind]
                if self.reversed_is_playable(ind):
                    playable_inds.append(ind)
        return playable_inds

    def reversed_image(self):
        all_reversed = []
        inds = []
        for ind in self.reversed_playable_inds():
            partition_rev = self.copy()
            partition_rev.reversed_bulgarian_solitaire_step(ind)
            all_reversed.append(partition_rev)
            inds.append(ind)
        return all_reversed, inds

    def mu_representation(self):
        n_parts = self.get_nparts()
        if n_parts == 1:
            mu = [self.parts[0]]
        else:
            mu = list()
            for i in range(n_parts - 1):
                mu.append(self.parts[i] - self.parts[i + 1])
            mu.append(self.parts[-1])
        return Mu(mu)

    # Returns true if this is a partition of the form 1 + 2 + ... + k.
    def is_tri(self):
        for i, part in enumerate(self.parts):
            if (i + 1 != part):
                return False
        return True


class PartitionAsNParts():
    def __init__(self, n, parts):
        # The integer of which this is a partition
        self.n = n

        self.nparts = [0] * n
        for i in parts:
            self.nparts[i - 1] += 1

    def to_key(self):
        # return tuple(self.nparts)
        out = []
        for i, i_nparts in enumerate(self.nparts):
            part_size = i + 1
            if i_nparts > 0:
                out.append(f"{part_size}^{i_nparts}")
        return ','.join(out)

    def bulgarian_solitaire_step(self):
        new_pile = sum(self.nparts)
        # new_pile = 0
        for i in range(self.n - 1):
            # new_pile += self.nparts[i]
            self.nparts[i] = self.nparts[i + 1]
        # new_pile += self.nparts[self.n - 1]
        self.nparts[self.n - 1] = 0
        self.nparts[new_pile - 1] += 1


class Mu():
    def __init__(self, mu_parts):
        self.parts = mu_parts

    def __str__(self):
        return str(self.parts)

    def __repr__(self):
        return str(self.parts)

    def ind1(self, ind):
        assert(ind >= 1), "Index must be 1, 2, 3, ..."
        if ind - 1 < len(self.parts):
            return self.parts[ind - 1]
        else:
            return 0




# if __name__ == "__main__":
#     n_steps = 0
#     p = Partition([500 * 1001])
#     tic = time.time()
#     # while not p.is_tri():
#     while n_steps < 500000:
#         p.bulgarian_solitaire_step()
#         n_steps += 1
#     toc = time.time()
#     # print(f"{n_steps} steps")
#     # print(f"{toc - tic} sec")
#     # print(f"{(toc - tic) / n_steps} sec/step")

#    /** */
#    public final Partition[] bulgSolReversedArr() {
#       ArrayList<Partition> outList = new ArrayList<Partition>();
#       bulgSolReversed(outList);
#       final int len = outList.size();
#       Partition[] out = new Partition[len];
#       for (int i=0; i<len; i++)
#          out[i] = (Partition) outList.get(i);
#       return out;
#    }

#    /**
#     * Adds to the specified list all partitions that lead to this in Bulgarian
#     * Solitaire.
#     * @param list The list to which the partitions are added
#     */
#    public final void bulgSolReversed(ArrayList<Partition> list) {
#       final int nParts = getNParts();
#       boolean done = false;
#       int ind = 0;
#       int prevPart;
#       while (!done && ind<=nParts-1) {
#          final int part = parts.get(ind);
#          done = (part<nParts-1);
#          if (done)
#             break;
#          if (ind>0) {
#             prevPart = parts.get(ind-1);
#             if (prevPart == part) {
#                ind++;
#                continue;
#             }
#          }
#          if (!done) {
#             Partition p = new Partition();
#             for (int i=0; i<nParts; i++)
#                if (i != ind)
#                   p.parts.add(parts.get(i)+1);
#             for (int i=0; i<part-nParts+1; i++)
#                p.parts.add(1);
#             list.add(p);
#             ind++;
#          }
#       }

#    }

   # /** Returns the  string [5^2,3,2^3,1] for the partition [5,5,3,2,2,2,1]. */
#    public String toCompactString() {
#       StringBuilder buf = new StringBuilder();
#       buf.append("[");
#       final int nParts = parts.size();
#       if (nParts == 1)
#          return "[" + parts.get(0) + "]";
#       int cnt = 1;
#       for (int i=1; i<nParts; i++) {
#          final int prevPart = parts.get(i-1);
#          final int thisPart = parts.get(i);
#          if (prevPart == thisPart) {
#             cnt++;
#             if (i == nParts-1)
#                buf = buf.append(prevPart).append("^").append(cnt);
#          }
#          else {
#             if (cnt>1)
#                buf = buf.append(prevPart).append("^").append(cnt);
#             else
#                buf = buf.append(prevPart);
#             if (i == nParts-1)
#                buf = buf.append(",").append(thisPart);
#             cnt = 1;
#             if (i<nParts-1)
#                buf = buf.append(",");
#          }

#       }
#       buf = buf.append("]");
#       return buf.toString();
#   }


#    public boolean equals(Partition that) {
#      	IntList thisIntList = this.getParts();
#   	   IntList thatIntList = that.getParts();
#   	   if (thisIntList.size() != thatIntList.size())
#   		   return false;
#   	   int size = thisIntList.size();
#   	   for (int i=0; i<size; i++)
#      		if (thisIntList.get(i) != thatIntList.get(i))
#      			return false;
#       return true;
#   }

#    public static Partition union(Partition p1, Partition p2) {
#       IntList s;
#       IntList l;
#       if (p1.getNParts() >= p2.getNParts()) {
#          l = p1.getParts();
#          s = p2.getParts();
#       }
#       else {
#          l = p2.getParts();
#          s = p1.getParts();
#       }
#       IntList out = new IntList();
#       for (int i=0; i<l.size(); i++) {
#          if (i<s.size())
#             out.add(Math.max(s.get(i), l.get(i)));
#          else
#             out.add(l.get(i));
#       }
#       return new Partition(out);
#    }


#    /** */
#    public boolean isGardenOfEden() {
#       return (getNParts() > 1+getLargestPart());
#    }

def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield PartitionAsNParts(n, a[:k + 2])
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield PartitionAsNParts(n, a[:k + 1])


class PartitionGenerator():
    def __init__(self, n):
        self.n = n

        # Helper
        self.x = [1] * n
        self.x[0] = n

        # The number of parts in the (previously computed) partition
        self.m = 1

        # Helper index to x
        self.h = 1

        # True before first partition has been computed
        self.first_time = True

    def get_next(self, return_partition_as_nparts=True):
        '''
        Uses the algorithm "ZS1" in Antoine Zoghbiu and Ivan Stojmenovic, "Fast
        algorithms for generating integer partitions", Intern. J. Computer Math.,
        Vol- 70. pp. 319 332.
        '''
        if self.first_time:
            self.first_time = False
            # return Partition([self.n], do_sort=False)  # First partition is n itself
            if return_partition_as_nparts:
                return PartitionAsNParts(self.n, [self.n])  # First partition is n itself
            else:
                return Partition([self.n], do_sort=False)

        if self.x[self.h - 1] == 2:
            self.m += 1
            self.x[self.h - 1] = 1
            self.h -= 1
        else:
            r = self.x[self.h - 1] - 1
            t = self.m - self.h + 1
            self.x[self.h - 1] = r
            while (t >= r):
                self.h += 1
                self.x[self.h - 1] = r
                t = t - r

            if t == 0:
                self.m = self.h
            else:
                self.m = self.h + 1
                if t > 1:
                    self.h += 1
                    self.x[self.h - 1] = t

        # x[0], ..., x[m-1] is the partition.
        parts = self.x[0: self.m]

        # parts.reverse()  # Reverse since Partition stores parts in increasing order
        # return Partition(parts, do_sort=False)
        if return_partition_as_nparts:
            return PartitionAsNParts(self.n, parts)
        else:
            return Partition(parts, do_sort=False)


    # Returns true if there are more partitions to compute. */
    def has_next(self):
        if self.first_time:
            return True
        return (self.x[0] != 1)


def get_mu_prim_paper(_lambda, lambda_prim, mu, j):
    lambda_1 = _lambda.ind1(1)
    lambda_j = _lambda.ind1(j)
    mu_prim = list()
    for i in range(1, 1 + lambda_prim.get_nparts()):
        if j == 1:
            if i == lambda_1:
                mu_prim_i = mu.ind1(i + 1) + 1
            else:
                mu_prim_i = mu.ind1(i + 1)
        elif j >= 2 and lambda_j != j - 1:
            if i < j - 1:
                mu_prim_i = mu.ind1(i)
            elif i == j - 1:
                mu_prim_i = mu.ind1(j - 1) + mu.ind1(j)  # Changed
            elif i >= j and i != lambda_j:
                mu_prim_i = mu.ind1(i + 1)
            elif i >= j and i == lambda_j:
                mu_prim_i = mu.ind1(i + 1) + 1
            else:
                assert(False), "Internal error"
        elif j >= 2 and lambda_j == j - 1:
            if i < j - 1:
                mu_prim_i = mu.ind1(i)
            elif i == j - 1:
                mu_prim_i = mu.ind1(i - 1) + mu.ind1(i) + 1
            elif i >= j:
                mu_prim_i = mu.ind1(i + 1)
            else:
                assert(False), "Internal error"
        else:
            assert(False), "Internal error"

        mu_prim.append(mu_prim_i)
    return mu_prim


def get_mu_prim_mine(_lambda, lambda_prim, mu, j):
    lambda_j = _lambda.ind1(j)
    mu_prim = list()
    for i in range(1, 1 + lambda_prim.get_nparts()):
        if i < j - 1:
            mu_prim_i = mu.ind1(i)
        elif i == j - 1:
            mu_prim_i = mu.ind1(i) + mu.ind1(i + 1)
        elif i > j - 1 and i != lambda_j:
            mu_prim_i = mu.ind1(i + 1)
        elif i > j - 1 and i == lambda_j:
            mu_prim_i = mu.ind1(i + 1) + 1
        else:
            assert(False), "Internal error"
        mu_prim.append(mu_prim_i)
    return mu_prim


if __name__ == "__main__":
    N = 30
    for n in range(1, N + 1):
        n_partitions = 0
        pg = PartitionGenerator(n)
        while (pg.has_next()):
            p = pg.get_next(return_partition_as_nparts=False)
            _lambda = p.copy()

            # print(p)
            n_partitions += 1

            mu = _lambda.mu_representation()
            n_parts = _lambda.get_nparts()
            for j in range(1, 1 + n_parts):
                if _lambda.reversed_is_playable(j):
                    tmp = _lambda.copy()
                    _lambda.reversed_bulgarian_solitaire_step(j)  # This changes _lambda
                    lambda_prim = _lambda.copy()
                    _lambda = tmp  # Restore _lambda

                    mu_prim = lambda_prim.mu_representation()
                    mu_prim_paper = get_mu_prim_paper(_lambda, lambda_prim, mu, j)
                    mu_prim_mine = get_mu_prim_mine(_lambda, lambda_prim, mu, j)

                    print(f"lambda={_lambda}")
                    print(f"mu={mu}")
                    print(f"j={j}")
                    print(f"lambda_prim={lambda_prim}")
                    print(f"    mu_prim={mu_prim}")
                    print(f"    mu_prim_paper={mu_prim_paper}")
                    print(f"    mu_prim_mine={mu_prim_mine}")
                    if mu_prim.parts != mu_prim_paper or mu_prim.parts != mu_prim_mine:
                        print("DIFF!")
                        input()
                    # print(f"mu_prim_mine={mu_prim_mine}")

        print(f"{n_partitions} partitions of {n}")
