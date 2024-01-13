import bisect
import time


class Partition():
    def __init__(self, parts, do_sort=True):
        # The integer of which this is a partition
        self.n = sum(parts)
    
        # The list of parts
        self.parts = parts

        if do_sort:
            self.sort_and_remove_nonpos_parts()
  
    def sort_and_remove_nonpos_parts(self):
        if (len(self.parts) == 0):
            return
        self.parts.sort()
        done = False
        while not done:
            last_ind = len(self.parts) - 1
            done = (self.parts[last_ind] > 0)
            if not done:
                self.parts.pop()

    def to_tuple(self):
        # return str(self.parts)
        return tuple(self.parts)

    def bulgarian_solitaire_step(self):
        n_piles = len(self.parts)
        for i in reversed(range(len(self.parts))):
            self.parts[i] -= 1
            if self.parts[i] == 0:
                self.parts = self.parts[(i + 1):]
                break
        bisect.insort_left(self.parts, n_piles)

    # Returns true if this is a partition of the form 1 + 2 + ... + k.
    def is_tri(self):
        for i, part in enumerate(self.parts):
            if (i + 1 != part):
                return False
        return True

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

    def get_next(self):
        '''
        Uses the algorithm "ZS1" in Antoine Zoghbiu and Ivan Stojmenovic, "Fast
        algorithms for generating integer partitions", Intern. J. Computer Math.,
        Vol- 70. pp. 319 332.
        '''
        if self.first_time:
            self.first_time = False
            return Partition([self.n], do_sort=False)  # First partition is n itself
        
        if self.x[self.h - 1] == 2:
            self.m += 1
            self.x[self.h-1] = 1
            self.h -= 1
        else:
            r = self.x[self.h - 1] - 1
            t = self.m - self.h + 1
            self.x[self.h - 1] = r
            while (t >= r):
                self.h = self.h + 1
                self.x[self.h - 1] = r
                t = t - r

            if t==0:
                self.m = self.h
            else:
                self.m = self.h + 1
                if t > 1:
                    self.h = self.h + 1
                    self.x[self.h - 1] = t
            
        # x[0], ..., x[m-1] is the partition. 
        parts = self.x[0: self.m]
        parts.reverse()  # Reverse since Partition stores parts in increasing order
        return Partition(parts, do_sort=False)

    # Returns true if there are more partitions to compute. */
    def has_next(self):
        if self.first_time:
            return True
        return (self.x[0] != 1)


if __name__ == "__main__":
    for n in range(1, 100):
        n_partitions = 0
        pg = PartitionGenerator(n)
        while (pg.has_next()):
            p = pg.get_next()
            n_partitions += 1
            # print(p.parts)
        print(f"{n_partitions} partitions of {n}")
