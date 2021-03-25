#
# bard511 example
# Using implicit index of lower level
# Using numpy/scipy data
#
import numpy as np
from scipy.sparse import coo_matrix
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = np.array([0])
    U.c[U] = np.array([1])
    U.c[L] = np.array([-4])

    L.x.lower_bounds = np.array([0])
    L.c[L] = np.array([1])

    L.A[U] = coo_matrix((np.array([-1, -2, 2, 3]),
                        (np.array([0, 1, 2, 3]),
                         np.array([0, 0, 0, 0]))))
    L.A[L] = coo_matrix((np.array([-1, 1, 1, -2]),
                        (np.array([0, 1, 2, 3]),
                         np.array([0, 0, 0, 0]))))
    L.b = np.array([-3, 0, 12, 4])

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = Solver('pao.mpr.REG')
    res = opt.solve(M, tee=False)
    #M.print()
    print("PAO")
    print(res)
