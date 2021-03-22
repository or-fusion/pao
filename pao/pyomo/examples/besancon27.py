# Example 2.7 from
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)
#
# Optimal solution: (x,v) = (0,1)
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.x = pe.Var(bounds=(0,None))
    M.v = pe.Var()

    M.o = pe.Objective(expr=M.x, sense=pe.minimize)
    M.c = pe.Constraint(expr= M.v >= 1 - M.x/10)

    M.L = SubModel(fixed=M.x)
    M.L.o = pe.Objective(expr=M.v, sense=pe.maximize)
    M.L.c = pe.Constraint(expr= 1 + M.x/10 >= M.v)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    #M.pprint()
    opt = Solver("pao.pyomo.FA")
    opt.solve(M)
    print(pe.value(M.x))
    print(pe.value(M.v))
