#
# Toy Example 3 from 
#
# "A projection-based reformulation and decomposition algorithm for global optimization 
#  of a class of mixed integer bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You
#
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxR=1, nxZ=1)
    U.x.lower_bounds = [0,0]

    L = U.add_lower(nxR=1, nxZ=1)
    L.x.lower_bounds = [0, 0]

    U.minimize = True
    U.c[U] = [20, -38]
    U.c[L] = [1, 42]

    U.A[U] = [[7, 5],
              [6, 9]]
    U.A[L] = [[0, 7],
              [10, 2]]
    U.b = [62, 117]

    L.maximize = True
    L.c[L] = [39, 27]

    L.A[U] = [[8, 0],
              [9, 0]]
    L.A[L] = [[2, 8], 
              [2, 1]]
    L.b = [53, 28]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = Solver('pao.mpr.PCCG')
    opt.solve(M)
    M.print()
