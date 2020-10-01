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
    U.xR.lower_bounds = [0]
    U.xZ.lower_bounds = [0]

    L = M.add_lower(nxR=1, nxZ=1)
    L.xR.lower_bounds = [0]
    L.xZ.lower_bounds = [0]
    L.minimize = False

    U.c.U.xR = [20]
    U.c.U.xZ = [-38]
    U.c.L.xR = [1]
    U.c.L.xZ = [42]

    U.A.U.xR = [[7], [6]]
    U.A.U.xZ = [[5], [9]]
    U.A.L.xR = [[0], [10]]
    U.A.L.xZ = [[7], [2]]
    U.b = [62,117]

    L.c.L.xR = [39]
    L.c.L.xZ = [27]

    L.A.U.xR = [[8], [9]]
    #L.A.U.xZ = [[-3], [3]]
    L.A.L.xR = [[2], [2]]
    L.A.L.xZ = [[8], [1]]
    L.b = [53,28]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
