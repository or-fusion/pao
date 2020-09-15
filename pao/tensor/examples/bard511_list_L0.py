#
# bard511 example
# Using explicit index of lower level
# Using Python list data
#
from pao.tensor import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.xR.lower_bounds = [0]
    U.c.U.xR = [1]
    U.c.L[0].xR = [-4]

    L = M.add_lower(nxR=1)
    L[0].xR.lower_bounds = [0]
    L[0].c.L[0].xR = [1]

    L[0].A.U.xR = [(0,0,-1), (1,0,-2), (2,0,2), (3,0,3)]
    L[0].A.L[0].xR = [(0,0,-1),  (1,0,1),  (2,0,1), (3,0,-2)]

    L[0].b = [-3, 0, 12, 4]

    return M


if __name__ == "__main__":
    M = create()
    opt = LinearBilevelSolver('pao.bilevel.blp_global')
    opt.solve(M)
    M.print()
