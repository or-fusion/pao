#
# Example from
# Moore, J. and J. Bard 1990.
# The mixed integer linear bilevel programming problem.
# Operations Research 38(5), 911–921.
#
from pao.tensor import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxZ=1)
    U.c.U.xZ = [-1]
    U.c.L.xZ = [-10]

    L = M.add_lower(nxZ=1)
    L.c.L.xZ = [1]

    L.A.U.xZ = [(0,0,-25),
                (1,0,1),
                (2,0,2),
                (3,0,-2)]
    L.A.L.xZ = [(0,0,20),
                (1,0,2),
                (2,0,-1),
                (3,0,-10)]
    L.b = [30,10,15,-15]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = LinearBilevelSolver('pao.lbp.FA')
    opt.solve(M)
    M.print()
