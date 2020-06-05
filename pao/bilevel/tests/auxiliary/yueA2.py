#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pao.bilevel import *

# Yue et al... Toy Example 2

def pyomo_create_model():
    M = ConcreteModel()
    M.yu=Var(within=NonNegativeIntegers,bounds=(0,10000))
    M.yl=Var(within=NonNegativeIntegers,bounds=(0,10000))
    M.c1 = Constraint(expr=-2*M.yu+3*M.yl <= 12)
    M.c2 = Constraint(expr= M.yu+M.yl <= 14)
    M.o = Objective(expr=-M.yu-2*M.yl)
    
    M.sub = SubModel(fixed=(M.yu))
    M.sub.o  = Objective(expr=M.yl, sense=maximize)
    M.sub.c3 = Constraint(expr=-3*M.yu + M.yl <= -3)
    M.sub.c4 = Constraint(expr= 3*M.yu + M.yl <= 30)

    return M
