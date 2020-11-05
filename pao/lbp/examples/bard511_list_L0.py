#
# bard511 example
# Using explicit index of lower level
# Using Python list data
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.x.lower_bounds = [0]
    U.c.U.x = [1]
    U.c.L[0].x = [-4]

    L = M.add_lower(nxR=1)
    L[0].x.lower_bounds = [0]
    L[0].c.L[0].x = [1]

    L[0].A.U.x = [[-1], [-2], [2], [3]]
    L[0].A.L[0].x = [[-1], [1], [1], [-2]]

    L[0].b = [-3, 0, 12, 4]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.bilevel.blp_global')
    opt.solve(M)
    M.print()
