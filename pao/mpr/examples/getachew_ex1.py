#
# Example from Getachew, Mersha and Dempe, 2005.
# Constraints in the upper-level
#
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.c[U] = [-1]
    U.c[L] = [-2]

    U.A[U] = [[-2],
              [1]]
    U.A[L] = [[3],
              [1]]
    U.b = [12, 14]

    L.c[L] = [-1]

    L.A[U] = [[-3],
              [3]]
    L.A[L] = [[1], 
              [1]]
    L.b = [-3, 30]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = Solver('pao.mpr.FA')
    opt.solve(M, tee=True)
    M.print()
