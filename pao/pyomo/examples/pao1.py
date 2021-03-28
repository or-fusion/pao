# 
# Example used in the PAO documentation
#   Simple bilevel
#
# Optimal solution: (x,y,z) = (6,4,2)
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.x = pe.Var(bounds=(2,6))
    M.y = pe.Var()
    M.z = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.x + 3*M.z, sense=pe.minimize)
    M.c = pe.Constraint(expr= M.x + M.y == 10)

    M.L = SubModel(fixed=[M.x,M.y])
    M.L.o = pe.Objective(expr=M.z, sense=pe.maximize)
    M.L.c1 = pe.Constraint(expr= M.x + M.z <= 8)
    M.L.c2 = pe.Constraint(expr= M.x + 4*M.z >= 8)
    M.L.c3 = pe.Constraint(expr= M.x + 2*M.z <= 13)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    #M.pprint()
    opt = Solver("pao.pyomo.FA")
    opt.solve(M)
    print(pe.value(M.x))
    print(pe.value(M.y))
    print(pe.value(M.z))
