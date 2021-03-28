# 
# Example used in the PAO documentation
#   Simple bilevel
#
# Optimal solution: (x,y,z) = (6,4,2)
#
import numpy as np
import pyomo.environ as pe
import pao.mpr


def create():
    M = pao.mpr.LinearMultilevelProblem()

    U = M.add_upper(nxR=2)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = [2, np.NINF]
    U.x.upper_bounds = [6, np.PINF]
    L.x.lower_bounds = [0]
    L.x.upper_bounds = [np.PINF]

    U.c[U] = [1, 0]
    U.c[L] = [3]

    L.c[L] = [1]
    L.maximize = True

    U.equalities = True
    U.A[U] = [[1, 1]]
    U.b = [10]

    L.A[U] = [[ 1, 0],
              [-1, 0],
              [ 1, 0]]
    L.A[L] = [[ 1],
              [-4],
              [ 2]]
    L.b = [8, -8, 13]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = pao.Solver("pao.mpr.FA")
    opt.solve(M)
    print(M.U.x.values)
    print(M.U.LL.x.values)
