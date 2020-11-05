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
    U.x.lower_bounds = [0]

    L = M.add_lower(nxZ=1)
    L.x.lower_bounds = [0]
    L.minimize = False

    U.c.U.x = [-1]
    U.c.L.x = [-2]

    U.A.U.x = [[-2], [1]]
    U.A.L.x = [[3], [1]]
    U.b = [12,14]

    L.c.L.x = [1]

    L.A.U.x = [[-3], [3]]
    L.A.L.x = [[1], [1]]
    L.b = [-3,30]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
