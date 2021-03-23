#
# Example from
# Moore, J. and J. Bard 1990.
# The mixed integer linear bilevel programming problem.
# Operations Research 38(5), 911–921.
#
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxZ=1)
    L = U.add_lower(nxZ=1)

    U.c[U] = [-1]
    U.c[L] = [-10]

    L.c[L] = [1]

    L.A[U] = [[-25],
              [1],
              [2],
              [-2]]
    L.A[L] = [[20],
              [2],
              [-1],
              [-10]]
    L.b = [30,10,15,-15]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.print()
    opt = Solver('pao.mpr.FA')
    opt.solve(M)
