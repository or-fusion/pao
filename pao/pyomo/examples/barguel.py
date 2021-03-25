#
# Simple interdiction problem solvable through
# lower level dualization and MPEC transformation
#
# Note that this is a quadratic bilevel problem
#
# This model has an optimal objective value of 0
# To see this, fix u and optimize the lower level
#
import pyomo.environ as pe
from pao.pyomo import SubModel


def create():
    M = pe.ConcreteModel()
    M.u = pe.Var(within=pe.Binary, initialize=1)
    M.x = pe.Var(bounds=(0,1)) # dual var for UB: v1
    M.y = pe.Var(bounds=(0,1)) # dual var for UB: v2

    # Note that the upper and lower level objectives
    # are the same.
    M.o = pe.Objective(expr=M.x + M.u)

    M.sub = SubModel(fixed=M.u)
    M.sub.o = pe.Objective(expr=M.x + M.u, sense=pe.maximize)
    M.sub.c = pe.Constraint(expr= M.x == M.y*M.u) # x - y*u = 0, dual var: u1

    return M


######################
# DUALIZED:
# min   v1 + v2 + u
# s.t.  u1 + v1 >= 1    : x
#       -u*u1 + v2 >= 0  : y
#       v1, v2, >= 0
#       u1 unconstrained
