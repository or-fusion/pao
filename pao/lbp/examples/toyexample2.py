#
# Toy Example 2 from 
#
# "A projection-based reformulation and decomposition algorithm for global optimization 
#  of a class of mixed integer bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxZ=1)
    U.minimize = True
    U.xZ.lower_bounds = [0]

    L = M.add_lower(nxZ=1)
    L.xZ.lower_bounds = [0]
    L.minimize = False

    U.c.U.xZ = [-1]
    U.c.L.xZ = [-2]

    U.A.U.xZ = [[-2], [1]]
    U.A.L.xZ = [[3], [1]]
    U.b = [12,14]

    L.c.L.xZ = [1]

    L.A.U.xZ = [[-3], [3]]
    L.A.L.xZ = [[1], [1]]
    L.b = [-3,30]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
