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

# Example Appendix A: Toy example 1 from
#
# A projection-base reformulation and decomposition
# algorithm for global optimization of a class of mixed integer
# bilevel linear programs
# By D. Yue, J. Gao, B. Zeng and F. You

def pyomo_create_model():
    M = ConcreteModel()
    M.y_u = Var(within=PositiveIntegers, bounds=(0,10000))
    M.y_l = Var(within=PositiveIntegers, bounds=(0,10000))
    M.o = Objective(expr=-M.y_u-10*M.y_l)
    
    M.sub = SubModel(fixed=(M.y_u))
    M.sub.o  = Objective(expr=-M.y_l, sense=maximize)
    M.sub.c1 = Constraint(expr=-25*M.y_u + 20*M.y_l <= 30)
    M.sub.c2 = Constraint(expr=M.y_u + 2*M.y_l <= 10)
    M.sub.c3 = Constraint(expr=2*M.y_u - M.y_l <= 15)
    M.sub.c4 = Constraint(expr=-2*M.y_u - 10*M.y_l <= -15)

    return M

