# 
# MIBS generalExample
#
# https://coral.ise.lehigh.edu/wp-content/uploads/2016/02/MibS_inputFile.pdf
#
# The optimal solution is C0=6, C1=5
#
import pyomo.environ as pe
from pao.pyomo import *


def create():
    M = pe.ConcreteModel()

    M.x = pe.Var(bounds=(0,10), within=pe.Integers)
    M.y = pe.Var(bounds=(0,5), within=pe.Integers)

    M.o = pe.Objective(expr=-M.x - 7*M.y, sense=pe.minimize)
    M.c1 = pe.Constraint(expr= -3*M.x + 2*M.y <= 12)
    M.c2 = pe.Constraint(expr=    M.x + 2*M.y <= 20)

    M.L = SubModel(fixed=M.x)
    M.L.o = pe.Objective(expr=M.y, sense=pe.minimize)
    M.L.c1 = pe.Constraint(expr=  2*M.x -   M.y <= 7)
    M.L.c2 = pe.Constraint(expr= -2*M.x + 4*M.y <= 16)

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    #M.pprint()
    opt = Solver("pao.pyomo.MIBS")
    results = opt.solve(M, tee=True)
    print(pe.value(M.x))
    print(pe.value(M.y))
