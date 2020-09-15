#
# Generalized bard511 example with two lower-levels
# Using explicit index of lower level
# Using numpy/scipy data
#
import numpy as np
from scipy.sparse import coo_matrix
from pao.tensor import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.xR.lower_bounds = np.array([0])
    U.c.U.xR = np.array([1])
    U.c.L[0].xR = np.array([-4])
    U.c.L[1].xR = np.array([-5])

    L = M.add_lower(nxR=1)
    L[0].xR.lower_bounds = np.array([0])
    L[0].c.L[0].xR = np.array([1])

    L[0].A.U.xR = coo_matrix((np.array([-1, -2, 2, 3]),
                          (np.array([0, 1, 2, 3]),
                           np.array([0, 0, 0, 0]))))
    L[0].A.L[0].xR = coo_matrix((np.array([-1, 1, 1, -2]),
                          (np.array([0, 1, 2, 3]),
                           np.array([0, 0, 0, 0]))))
    L[0].b = np.array([-3, 0, 12, 4])

    L = M.add_lower(nxR=1)
    L[1].xR.lower_bounds = np.array([-1])
    L[1].c.L[1].xR = np.array([2])

    L[1].A.U.xR = coo_matrix((np.array([-1, -2, 2, -3]),
                          (np.array([0, 1, 2, 3]),
                           np.array([0, 0, 0, 0]))))
    L[1].A.L[1].xR = coo_matrix((np.array([-1, 1, 1, 2]),
                          (np.array([0, 1, 2, 3]),
                           np.array([0, 0, 0, 0]))))
    L[1].b = np.array([-3, 0, 12, -4])

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.bilevel.blp_global')
    opt.solve(M)
    M.print()
