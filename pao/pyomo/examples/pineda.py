#
# Example from Pineda and Morales (2018), showing how
# bad estimates of bigMs give the wrong answer.
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.xR = pe.Var(bounds=(0,2))

    M.L = SubModel(fixed=(M.xR))
    M.L.xR = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.xR + M.L.xR, sense=pe.maximize)

    M.L.o = pe.Objective(expr=M.L.xR)
    M.L.c = pe.Constraint(expr=100*M.xR - M.L.xR <= 100)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    M.pprint()
