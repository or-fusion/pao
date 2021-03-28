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

    M.w = pe.Var([1,2], within=pe.Binary)
    M.x = pe.Var(bounds=(2,6))
    M.y = pe.Var()
    M.z = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.x + 3*M.z+5*M.w[1], sense=pe.minimize)
    M.c1 = pe.Constraint(expr= M.x + M.y == 10)
    M.c2 = pe.Constraint(expr= M.w[1] + M.w[2] >= 1)

    M.L = SubModel(fixed=[M.x,M.y,M.w])
    M.L.o = pe.Objective(expr=M.z, sense=pe.maximize)
    M.L.c1 = pe.Constraint(expr= M.x + M.w[1]*M.z <= 8)
    M.L.c2 = pe.Constraint(expr= M.x + 4*M.z >= 8)
    M.L.c3 = pe.Constraint(expr= M.x + 2*M.w[2]*M.z <= 13)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    #M.pprint()
    #M.pprint()
    opt = Solver("pao.pyomo.FA", linearize_bigm=100)
    opt.solve(M)
    #M.pprint()
    print("="*10)
    print(pe.value(M.x))
    print(pe.value(M.y))
    print(pe.value(M.z))
    print(pe.value(M.w[1]))
    print(pe.value(M.w[2]))
