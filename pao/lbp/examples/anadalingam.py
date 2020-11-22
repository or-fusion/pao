#
# Trilevel example 
#
# Anandalingam, G.: A mathematical programming model of decentralized multi-level systems. 
# J. Oper.Res. Soc.39(11), 1021–1033 (1988)
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    U.minimize = True

    L = U.add_lower(nxR=1)
    L.minimize = True

    B = L.add_lower(nxR=1)
    B.minimize = True

    U.x.lower_bounds = [0]
    L.x.lower_bounds = [0]
    B.x.lower_bounds = [0]
    B.x.upper_bounds = [0.5]

    U.c[U] = [-7]
    U.c[L] = [-3]
    U.c[B] = [4]

    L.c[L] = [-1]

    B.c[B] = [-1]

    B.inequalities = True
    B.A[U] = [[1], [ 1], [-1], [-1]]
    B.A[L] = [[1], [ 1], [-1], [ 1]]
    B.A[B] = [[1], [-1], [-1], [ 1]]
    B.b = [3,1,-1,1]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.PCCG')
    opt.solve(M)
    M.print()
