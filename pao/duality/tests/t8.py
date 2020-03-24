
from pyomo.environ import *


M = ConcreteModel()
M.u = Var(within = Binary, initialize=1)
M.x = Var(bounds=(0,1)) # dual var for UB: v1
M.y = Var(bounds=(-1,0)) # dual var for LB: v2
M.z = Var(bounds=(-1,1)) # dual var for LB: v3, dual var for UB: v4

M.sub = Block()
M.sub.o = Objective(expr=M.x + M.u, sense=maximize)
M.sub.c = Constraint(expr= M.x == M.z + M.y*M.u) # x -z - y*u = 0, dual var: u1
M.sub.C = Constraint(expr= M.u == 1)

model = M

######################
# DUALIZED:
# min:  - v1 + v2 + v3 - v4 + u
# s.t.  c + v1       <-= -1   : x_upper
#       -u*c + v2    >= 0     : y_lower
#       -c + v3 + v4 == 0     : y_lower 
#       v1, v4 <= 0
#       v2, v3 >= 0
#       c unconstrained

