# 
# Example used in the PAO documentation
#   Multiple lower-levels
#
# Optimal solution: (x,y,z) = (6,4,2)
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.x = pe.Var(bounds=(2,6))
    M.y = pe.Var()
    M.z = pe.Var([1,2], bounds=(0,None))

    M.o = pe.Objective(expr=M.x + 3*M.z[1]+3*M.z[2], sense=pe.minimize)
    M.c = pe.Constraint(expr= M.x + M.y == 10)

    M.L1 = SubModel(fixed=[M.x])
    M.L1.o = pe.Objective(expr=M.z[1], sense=pe.maximize)
    M.L1.c1 = pe.Constraint(expr= M.x + M.z[1] <= 8)
    M.L1.c2 = pe.Constraint(expr= M.x + 4*M.z[1] >= 8)
    M.L1.c3 = pe.Constraint(expr= M.x + 2*M.z[1] <= 13)

    M.L2 = SubModel(fixed=[M.y])
    M.L2.o = pe.Objective(expr=M.z[2], sense=pe.maximize)
    M.L2.c1 = pe.Constraint(expr= M.y + M.z[2] <= 8)
    M.L2.c2 = pe.Constraint(expr= M.y + 4*M.z[2] >= 8)
    M.L2.c3 = pe.Constraint(expr= M.y + 2*M.z[2] <= 13)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    #M.pprint()
    opt = Solver("pao.pyomo.FA")
    opt.solve(M)
    print(pe.value(M.x))
    print(pe.value(M.y))
    print(M.z[1].value, M.z[2].value)
