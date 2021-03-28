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
    M = pao.mpr.QuadraticMultilevelProblem()

    U = M.add_upper(nxR=2, nxB=2)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = [2, np.NINF, 0, 0]
    U.x.upper_bounds = [6, np.PINF, 1, 1]
    L.x.lower_bounds = [0]
    L.x.upper_bounds = [np.PINF]

    U.c[U] = [1, 0, 5, 0]
    U.c[L] = [3]

    L.c[L] = [1]
    L.maximize = True

    U.A[U] = [[ 1,  1,  0,  0],
              [-1, -1,  0,  0],
              [ 0,  0, -1, -1]
              ]
    U.b = [10, -10, -1]

    L.A[U] = [[ 1, 0, 0, 0],
              [-1, 0, 0, 0],
              [ 1, 0, 0, 0]]
    L.A[L] = [[ 0],
              [-4],
              [ 0]]
    L.Q[U,L] = (3,4,1), {(0,2,0):1, (2,3,0):2}
                
    L.b = [8, -8, 13]

    return M


if __name__ == "__main__":          #pragma: no cover
    qmr = create()
    lmr, soln = pao.mpr.linearize_bilinear_terms(qmr, 100)
    opt = pao.Solver("pao.mpr.FA")
    opt.solve(lmr)
    soln.copy(From=lmr, To=qmr)
    print(qmr.U.x.values)
    print(qmr.U.LL.x.values)
