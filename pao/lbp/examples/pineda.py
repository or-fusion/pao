#
# Example from Pineda and Morales (2018), showing how
# bad estimates of bigMs give the wrong answer.
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.minimize = False
    U.x.lower_bounds = [0]
    U.x.upper_bounds = [2]
    U.c.U.x = [1]
    U.c.L.x = [1]

    L = M.add_lower(nxR=1)
    L.x.lower_bounds = [0]
    L.c.L.x = [1]

    L.A.U.x = [[100]]
    L.A.L.x = [[-1]]
    L.b = [100]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.FA')
    opt.solve(M)
    M.print()
