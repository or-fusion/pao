#
# Simple interdiction problem solvable through
# lower level dualization and MPEC transformation
#
# Note that this is a quadratic bilevel problem
#
# This model has an optimal objective value of 0
# To see this, fix u and optimize the lower level
#
# DUALIZED:
# min   v1 + v2 + u
# s.t.  u1 + v1 >= 1    : x
#       -u*u1 + v2 >= 0  : y
#       v1, v2, >= 0
#       u1 unconstrained
#

from pao.mpr import *

def create():
    M = QuadraticMultilevelProblem()

    U = M.add_upper(nxB=1)
    L = U.add_lower(nxR=2)
    L.maximize = True

    L.x.lower_bounds=[0, 0]
    L.x.upper_bounds=[1, 1]

    U.c[U] = [1]
    U.c[L] = [1, 0]

    L.c[U] = [1]
    L.c[L] = [1, 0]

    L.A[L] = [[ 1, 0],
              [-1, 0]]
    L.Q[U,L] = (2,1,2), {(0,0,1):-1, (1,0,1):1}
    L.b    = [0, 0]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    m = linearize_bilinear_terms(M)
    m.print()

    opt = Solver('pao.mpr.FA')
    opt.solve(m, tee=True)
    m.print()
