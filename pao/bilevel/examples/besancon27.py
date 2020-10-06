# Example 2.7 from
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)
#
import pyomo.environ as pe
from pao.bilevel import *


def create():
    M = pe.ConcreteModel()

    M.xR = pe.Var(bounds=(0,None))

    M.L = SubModel(fixed=M.xR)
    M.L.xR = pe.Var()

    M.o = pe.Objective(expr=M.xR, sense=pe.minimize)
    M.c = pe.Constraint(expr= M.xR/10 + M.L.xR >= 1)

    M.o = pe.Objective(expr=M.L.xR, sense=pe.maximize)
    M.L.c = pe.Constraint(expr= -M.xR/10 + M.L.xR <= 1)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()
