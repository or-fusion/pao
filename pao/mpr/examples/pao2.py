# 
# Example used in the PAO documentation
#   Multiple lower-levels
#
# Optimal solution: (x,y,z) = (6,4,2)
#
import numpy as np
import pyomo.environ as pe
import pao.mpr


def create():
    M = pao.mpr.LinearMultilevelProblem()

    U = M.add_upper(nxR=2)
    L1 = U.add_lower(nxR=1)
    L2 = U.add_lower(nxR=1)

    U.x.lower_bounds = [2, np.NINF]
    U.x.upper_bounds = [6, np.PINF]
    U.c[U] = [1, 0]
    U.c[L1] = [3]
    U.c[L2] = [3]
    U.equalities = True
    U.A[U] = [[1, 1]]
    U.b = [10]

    L1.x.lower_bounds = [0]
    L1.x.upper_bounds = [np.PINF]
    L1.c[L1] = [1]
    L1.maximize = True
    L1.A[U] = [[ 1, 0],
               [-1, 0],
               [ 1, 0]]
    L1.A[L1] = [[ 1],
                [-4],
                [ 2]]
    L1.b = [8, -8, 13]

    L2.x.lower_bounds = [0]
    L2.x.upper_bounds = [np.PINF]
    L2.c[L2] = [1]
    L2.maximize = True
    L2.A[U] = [[0,  1],
               [0, -1],
               [0,  1]]
    L2.A[L2] = [[ 1],
                [-4],
                [ 2]]
    L2.b = [8, -8, 13]

    return M

if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = pao.Solver("pao.mpr.FA")
    opt.solve(M)
    for L in M.levels():
        print(L.name, L.x.values)
