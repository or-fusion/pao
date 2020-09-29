#
# Toy Example 1 from 
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
    U.c.L.xZ = [-10]

    L.c.L.xZ = [-1]

    L.A.U.xZ = [[-25], [1], [2], [-2]]

    L.A.L.xZ = [[20], [2], [-1], [-10]]

    L.b = [30,10,15,-15]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
