#
# Simple quadratic example
#
import numpy as np
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1, nxB=1)
    U.lower_bounds = [0,0]

    L = U.add_lower(nxR=5)
    L.lower_bounds = [1,        -100,   np.NINF, 3, 0]
    L.upper_bounds = [np.PINF,     2,   np.PINF, 4, np.PINF]

    U.c[U] = [ 1, 0]
    U.c[L] = [-4, 0, 0, 0]


    L.c[U] = [0, 11]
    L.c[L] = [0,  1, 9, 0]

    L.A[U] = [[ 1, 0],
              [-2, 0],
              [ 4, 0],
              [ 3, 0],
              [-3, 0]]
    L.A[L] = [[ 5, 0, 0,  0,  1],
              [-6, 0, 0, 10, -1],
              [ 8, 0, 0,  0,  0],
              [ 7, 0, 0,  0,  1],
              [-7, 0, 0,  0, -1]]
    L.b = [19, -20, 32, 28, -22]
        
    L.c[U,L] = [[ 0, 0, 0, 0, 0],
                [12, 0, 0, 0, 0]]
    #L.c[U,L] = {(1,0):12}

    L.A[U,L] = [ [[ 0,  0,  0, 0, 0],
                  [13,  0,  0, 0, 0]],
                 [[ 0,  0,  0, 0, 0],
                  [ 0,-14,  0, 0, 0]],
                 [[ 0,  0,  0, 0, 0],
                  [ 0,  0,  0,15, 0]],
                 [[ 0,  0,  0, 0, 0],
                  [ 0,  0, 16, 0, 0]],
                 [[ 0,  0,  0, 0, 0],
                  [ 0,  0,-16, 0, 0]]
               ]
    #L.A[U,L] = {0: {(1,0):13},
    #            1: {(1,1):-14},
    #            2: {(1,3):15},
    #            3: {(1,2):16},
    #            4: {(1,2):-16}}

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.PCCG')
    opt.solve(M)
    M.print()

