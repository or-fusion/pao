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
    U.x.lower_bounds = [0]

    L = U.add_lower(nxZ=1)
    L.x.lower_bounds = [0]

    U.minimize = True
    U.c[U] = [-1]
    U.c[L] = [-10]

    L.minimize = False
    L.c[L] = [-1]

    L.A[U] = [[-25], [1], [2], [-2]]
    L.A[L] = [[20], [2], [-1], [-10]]
    L.b = [30,10,15,-15]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
