#
# Example from Pineda and Morales (2018), showing how
# bad estimates of bigMs give the wrong answer.
#
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.maximize = True
    U.x.lower_bounds = [0]
    U.x.upper_bounds = [2]
    U.c[U] = [1]
    U.c[L] = [1]

    L.x.lower_bounds = [0]
    L.c[L] = [1]

    L.A[U] = [[100]]
    L.A[L] = [[-1]]
    L.b = [100]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = Solver('pao.mpr.FA')
    opt.solve(M)
    M.print()
