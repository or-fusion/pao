#
# Example from Pineda and Morales (2018), showing how
# bad estimates of bigMs give the wrong answer.
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.minimize = False
    U.xR.lower_bounds = [0]
    U.xR.upper_bounds = [2]
    U.c.U.xR = [1]
    U.c.L.xR = [1]

    L = M.add_lower(nxR=1)
    L.xR.lower_bounds = [0]
    L.c.L.xR = [1]

    L.A.U.xR = [(0,0,100)]
    L.A.L.xR = [(0,0,-1)]
    L.b = [100]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.lbp.FA')
    opt.solve(M)
    M.print()
