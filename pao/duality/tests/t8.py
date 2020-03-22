
from pyomo.environ import *


M = ConcreteModel()
M.u = Var(within = Binary, initialize=1)
M.x = Var(bounds=(0,1)) # dual var for UB: v1
M.y = Var(bounds=(0,1)) # dual var for UB: v2

#M.sub = SubModel(fixed=M.x)
M.sub = Block()
M.sub.o = Objective(expr=M.x + M.u, sense=maximize)
M.sub.c = Constraint(expr= M.x == M.y*M.u) # x - y*u = 0, dual var: u1

model = M

######################
# DUALIZED:
# min   v1 + v2 + u
# s.t.  u1 + v1 >= 1    : x
#       -u*u1 + v2 >= 0  : y
#       v1, v2, >= 0
#       u1 unconstrained

