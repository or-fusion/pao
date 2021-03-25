#
# Toy Example 3 from 
#
# "A projection-based reformulation and decomposition algorithm for global optimization 
#  of a class of mixed integer bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.xR = pe.Var(bounds=(0,None))
    M.xZ = pe.Var(bounds=(0,None), within=pe.Integers)

    M.L = SubModel(fixed=(M.xR, M.xZ))
    M.L.xR = pe.Var(bounds=(0,None))
    M.L.xZ = pe.Var(bounds=(0,None), within=pe.Integers)

    M.o = pe.Objective(expr=20*M.xR - 38*M.xZ + M.L.xR + 42*M.L.xZ, sense=pe.minimize)
    M.c1 = pe.Constraint(expr=7*M.xR + 5*M.xZ +             7*M.L.xZ <= 62)
    M.c2 = pe.Constraint(expr=6*M.xR + 9*M.xZ + 10*M.L.xR + 2*M.L.xZ <= 117)

    M.L.o = pe.Objective(expr=39*M.L.xR + 27*M.L.xZ, sense=pe.maximize)
    M.L.c1 = pe.Constraint(expr=8*M.xR + 2*M.L.xR + 8*M.L.xZ <= 53)
    M.L.c2 = pe.Constraint(expr=9*M.xR + 2*M.L.xR + 1*M.L.xZ <= 28)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()

    with Solver('pao.pyomo.PCCG') as opt:
        results = opt.solve(M, tee=True, time_limit=10, quiet=False, maxit=100)

    print(results)
    M.display()

