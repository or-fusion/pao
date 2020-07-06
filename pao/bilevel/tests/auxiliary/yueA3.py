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

# Example Appendix C: Toy example 3 from
#
# A projection-base reformulation and decomposition
# algorithm for global optimization of a class of mixed integer
# bilevel linear programs
# By D. Yue, J. Gao, B. Zeng and F. You

def pyomo_create_model():
    M = ConcreteModel()
    M.xu=Var(within=NonNegativeReals,bounds=(0,10000))
    M.xl=Var(within=NonNegativeReals,bounds=(0,10000))
    M.yu=Var(within=NonNegativeIntegers,bounds=(0,10000))
    M.yl=Var(within=NonNegativeIntegers,bounds=(0,10000))
    M.c1 = Constraint(expr=7*M.yu + 5*M.xl + 7*M.yl <= 62)
    M.c2 = Constraint(expr= 6*M.xu + 9*M.yu + 10*M.xl + 2*M.yl <= 117)
    M.o = Objective(expr=20*M.xu - 38*M.yu + M.xl + 42*M.yl)
    
    M.sub = SubModel(fixed=(M.xu,M.yu))
    M.sub.o  = Objective(expr=39*M.xl+27*M.yl, sense=maximize)
    M.sub.c3 = Constraint(expr=8*M.xu + 2*M.xl + 8*M.yl <= 53)
    M.sub.c4 = Constraint(expr=9*M.xu + 2*M.xl + M.yl <= 28)

    return M

