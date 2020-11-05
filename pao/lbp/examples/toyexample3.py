#
# Toy Example 3 from 
#
# "A projection-based reformulation and decomposition algorithm for global optimization 
#  of a class of mixed integer bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1, nxZ=1)
    U.minimize = True
    U.x.lower_bounds = [0,0]

    L = M.add_lower(nxR=1, nxZ=1)
    L.x.lower_bounds = [0, 0]
    L.minimize = False

    U.c.U.x = [20, -38]
    U.c.L.x = [1, 42]

    U.A.U.x = [[7, 5],
               [6, 9]]
    U.A.L.x = [[0, 7],
               [10, 2]]
    U.b = [62, 117]

    L.c.L.x = [39, 27]

    L.A.U.x = [[8, 0],
               [9, 0]]
    L.A.L.x = [[2, 8], 
               [2, 1]]
    L.b = [53, 28]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
