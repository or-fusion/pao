#
# Example from
# Moore, J. and J. Bard 1990.
# The mixed integer linear bilevel programming problem.
# Operations Research 38(5), 911–921.
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.xZ = pe.Var(within=pe.Integers)

    M.L = SubModel(fixed=(M.xZ))
    M.L.xZ = pe.Var(within=pe.Integers)

    M.o = pe.Objective(expr=-M.xZ - 10*M.L.xZ)

    M.L.o = pe.Objective(expr=M.L.xZ)
    M.L.c1 = pe.Constraint(expr= -25*M.xZ + 20*M.L.xZ <=  30)
    M.L.c2 = pe.Constraint(expr=     M.xZ +  2*M.L.xZ <=  10)
    M.L.c3 = pe.Constraint(expr=   2*M.xZ -    M.L.xZ <=  15)
    M.L.c4 = pe.Constraint(expr=  -2*M.xZ - 10*M.L.xZ <= -15)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()
