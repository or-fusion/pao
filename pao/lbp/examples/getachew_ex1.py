#
# Example from Getachew, Mersha and Dempe, 2005.
# Constraints in the upper-level
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.c.U.xR = [-1]
    U.c.L.xR = [-2]

    U.A.U.xR = [[-2],
                [1]]
    U.A.L.xR = [[3],
                [1]]
    U.b = [12, 14]

    L = M.add_lower(nxR=1)
    L.c.L.xR = [-1]

    L.A.U.xR = [[-3],
                [3]]
    L.A.L.xR = [[1], 
                [1]]
    L.b = [-3, 30]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.FA')
    opt.solve(M, tee=True)
    M.print()
