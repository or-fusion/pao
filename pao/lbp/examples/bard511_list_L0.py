#
# bard511 example
# Using explicit index of lower level
# Using Python list data
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = [0]
    U.c[U] = [1]
    U.c[L] = [-4]

    L.x.lower_bounds = [0]
    L.c[L] = [1]

    L.A[U] = [[-1], [-2], [2], [3]]
    L.A[L] = [[-1], [1], [1], [-2]]

    L.b = [-3, 0, 12, 4]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.bilevel.blp_global')
    opt.solve(M)
    M.print()
