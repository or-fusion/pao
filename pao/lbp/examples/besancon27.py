# Example 2.7 from
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    # Variables
    U = M.add_upper(nxR=1)
    U.x.lower_bounds = [0]

    L = M.add_lower(nxR=1)

    # Objectives
    U.c.U.x = [1]

    L.c.L.x = [1]
    L.minimize = False

    # Constraints
    U.A.U.x = [[-1/10]]
    U.A.L.x = [[-1]]
    U.b = [-1]

    L.A.U.x = [[-1/10]]
    L.A.L.x = [[1]]
    L.b = [1]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.FA')
    opt.solve(M)
    M.print()
