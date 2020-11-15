# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:00:06 2020

@author: spunlag
"""
from pyomo.environ import *
from pao.bilevel import *
from pao.bilevel.solvers.solver5 import BilevelSolver5

M = ConcreteModel()
M.x=Var(within=NonNegativeReals,bounds=(0,10000))
M.v=Var(within=NonNegativeReals,bounds=(0,10000))
M.c1 = Constraint(expr=-M.x+4*M.v <= 11)
M.c2 = Constraint(expr= M.x+2*M.v <= 13)
M.o = Objective(expr=M.x-10*M.v)
    
M.sub = SubModel(fixed=(M.x))
#M.sub.o  = Objective(expr=-M.v, sense=maximize)
M.sub.o  = Objective(expr=M.v, sense=minimize)
M.sub.c3 = Constraint(expr=-2*M.x - M.v <= -5)
M.sub.c4 = Constraint(expr= 5*M.x - 4*M.v <= 30)


#opt=BilevelSolver5()
opt=SolverFactory('pao.bilevel.norvep')
opt.options.solver="gurobi"
opt.options.delta=1
opt.options.do_print=False
opt.solve(M)

M.x.pprint()
M.v.pprint()
print(f'{M.o.expr()}')