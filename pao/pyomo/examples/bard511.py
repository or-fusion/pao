# Example 5.1.1 from
#
# Practical Bilevel Optimization: Algorithms and Applications
#   Jonathan Bard

import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()
    M.x = pe.Var(bounds=(0,None))
    M.y = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.x - 4*M.y)

    M.L = SubModel(fixed=M.x)
    M.L.o = pe.Objective(expr=M.y)
    M.L.c1 = pe.Constraint(expr=   -M.x -   M.y <= -3)
    M.L.c2 = pe.Constraint(expr= -2*M.x +   M.y <=  0)
    M.L.c3 = pe.Constraint(expr=  2*M.x +   M.y <= 12)
    M.L.c4 = pe.Constraint(expr=  3*M.x - 2*M.y <=  4)
    #M.L.c4 = pe.Constraint(expr= -3*M.x + 2*M.y <= -4)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()
