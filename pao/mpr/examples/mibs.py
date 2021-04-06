# 
# MIBS generalExample
#
# https://coral.ise.lehigh.edu/wp-content/uploads/2016/02/MibS_inputFile.pdf
#
# The optimal solution is C0=6, C1=5
#
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxZ=1)
    L = U.add_lower(nxZ=1)

    # Variables
    U.x.lower_bounds = [0.0]
    U.x.upper_bounds = [10.0]
    L.x.lower_bounds = [0.0]
    L.x.upper_bounds = [5.0]

    # Objectives
    U.c[U] = [-1]
    U.c[L] = [-7]

    L.c[L] = [1]

    # Constraints
    U.A[U] = [[-3], [1]]
    U.A[L] = [[2], [2]]
    U.b = [12, 20]

    L.A[U] = [[2], [-2]]
    L.A[L] = [[-1], [4]]
    L.b = [7, 16]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.check()
    #M.print()
    opt = Solver('pao.mpr.MIBS')
    opt.solve(M)
    print(M.U.x.values)
    print(M.U.LL.x.values)
