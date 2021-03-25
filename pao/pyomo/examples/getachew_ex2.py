#
# Example from Getachew, Mersha and Dempe, 2005.
# Constraints in the lower-level
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.xR = pe.Var()

    M.L = SubModel(fixed=M.xR)
    M.L.xR = pe.Var()

    M.o = pe.Objective(expr=-M.xR - 2*M.L.xR)

    M.L.o = pe.Objective(expr=-M.L.xR)
    M.L.c1 = pe.Constraint(expr= -3*M.xR +   M.L.xR <= -3)
    M.L.c2 = pe.Constraint(expr=  3*M.xR +   M.L.xR <= 30)
    M.L.c3 = pe.Constraint(expr= -2*M.xR + 3*M.L.xR <= 12)
    M.L.c4 = pe.Constraint(expr=    M.xR +   M.L.xR <= 14)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()
