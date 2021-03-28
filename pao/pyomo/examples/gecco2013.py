# Example GECCO 2013 Tutorial
#
# Evolutionary Bilevel Optimization
#   A. Sinha, P. Malo, K. Deb
#   GECCO (2013)
#
# Optimal solution: (x,y) = (6,2)
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.x = pe.Var(bounds=(2,6))
    M.y = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.x + 3*M.y, sense=pe.minimize)

    M.L = SubModel(fixed=M.x)
    M.L.o = pe.Objective(expr=M.y, sense=pe.maximize)
    M.L.c1 = pe.Constraint(expr= M.x + M.y <= 8)
    M.L.c2 = pe.Constraint(expr= M.x + 4*M.y >= 8)
    M.L.c3 = pe.Constraint(expr= M.x + 2*M.y <= 13)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    #M.pprint()
    opt = Solver("pao.pyomo.FA")
    opt.solve(M)
    print(pe.value(M.x))
    print(pe.value(M.y))
