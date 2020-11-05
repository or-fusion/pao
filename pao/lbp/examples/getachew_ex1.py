#
# Example from Getachew, Mersha and Dempe, 2005.
# Constraints in the upper-level
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.c.U.x = [-1]
    U.c.L.x = [-2]

    U.A.U.x = [[-2],
               [1]]
    U.A.L.x = [[3],
               [1]]
    U.b = [12, 14]

    L = M.add_lower(nxR=1)
    L.c.L.x = [-1]

    L.A.U.x = [[-3],
               [3]]
    L.A.L.x = [[1], 
               [1]]
    L.b = [-3, 30]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.FA')
    opt.solve(M, tee=True)
    M.print()
