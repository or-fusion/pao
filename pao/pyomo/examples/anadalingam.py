#
# Trilevel example 
#
# Anandalingam, G.: A mathematical programming model of decentralized multi-level systems. 
# J. Oper.Res. Soc.39(11), 1021–1033 (1988)
#

import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()
    M.x1 = pe.Var(bounds=(0,None))
    M.x2 = pe.Var(bounds=(0,None))
    M.x3 = pe.Var(bounds=(0,0.5))

    M.L = SubModel(fixed=M.x1)

    M.L.B = SubModel(fixed=M.x2)

    M.o = pe.Objective(expr=-7*M.x1 - 3*M.x2 + 4*M.x3)

    M.L.o = pe.Objective(expr=-M.x2)
    M.L.B.o = pe.Objective(expr=-M.x3)

    M.L.B.c1 = pe.Constraint(expr=   M.x1 + M.x2 + M.x3 <= 3)
    M.L.B.c2 = pe.Constraint(expr=   M.x1 + M.x2 - M.x3 <= 1)
    M.L.B.c3 = pe.Constraint(expr=   M.x1 + M.x2 + M.x3 >= 1)
    M.L.B.c4 = pe.Constraint(expr= - M.x1 + M.x2 + M.x3 <= 1)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()
