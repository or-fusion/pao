#
# bard511 example
# Using implicit index of lower level
# Using Python list data
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.xR.lower_bounds = [0]
    U.c.U.xR = [1]
    U.c.L.xR = [-4]

    L = M.add_lower(nxR=1)
    L.xR.lower_bounds = [0]
    L.c.L.xR = [1]

    L.A.U.xR = [[-1], [-2], [2], [3]]
    L.A.L.xR = [[-1], [1], [1], [-2]]

    L.b = [-3, 0, 12, 4]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.lbp.FA')
    opt.solve(M)
    M.print()
