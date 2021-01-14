# Example 2.7 from
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    # Variables
    U.x.lower_bounds = [0]

    # Objectives
    U.c[U] = [1]

    L.c[L] = [1]
    L.minimize = False

    # Constraints
    U.A[U] = [[-1/10]]
    U.A[L] = [[-1]]
    U.b = [-1]

    L.A[U] = [[-1/10]]
    L.A[L] = [[1]]
    L.b = [1]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.FA')
    opt.solve(M)
    M.print()
