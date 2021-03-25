#
# Trilevel example 
#
# Anandalingam, G.: A mathematical programming model of decentralized multi-level systems. 
# J. Oper.Res. Soc.39(11), 1021–1033 (1988)
#
# This example illustrates how pao.mpr can model tri-level problems, though
# we don't have PAO solvers that support these yet.
#
from pao.mpr import *


def create():
    M = LinearMultilevelProblem()

    U = M.add_upper(nxR=1)
    U.x.lower_bounds = [0]

    L = U.add_lower(nxR=1)
    L.x.lower_bounds = [0]

    B = L.add_lower(nxR=1)
    B.x.lower_bounds = [0]
    B.x.upper_bounds = [0.5]

    U.minimize = True
    U.c[U] = [-7]
    U.c[L] = [-3]
    U.c[B] = [4]

    L.minimize = True
    L.c[L] = [-1]

    B.minimize = True
    B.c[B] = [-1]

    B.inequalities = True
    B.A[U] = [[1], [ 1], [-1], [-1]]
    B.A[L] = [[1], [ 1], [-1], [ 1]]
    B.A[B] = [[1], [-1], [-1], [ 1]]
    B.b = [3,1,-1,1]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = Solver('pao.mpr.PCCG')
    opt.solve(M)
    M.print()
