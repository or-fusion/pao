#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Simple interdiction problem solvable through
# lower level dualization and MPEC transformation

# This model has an optimal objective value of 0
# To see this, fix u and optimize the lower level

from pyomo.environ import *
from pao.bilevel import *

def pyomo_create_model():
    M = ConcreteModel()
    M.u = Var(within = Binary)
    M.x = Var(bounds=(0,1))
    M.y = Var(bounds=(0,1))
    # Note that the upper and lower level objectives
    # are the same. I would not actually use M.o, but
    # I am creating it here for transparency.
    M.o = Objective(expr=M.x + M.u)

    #M.sub = SubModel(fixed=M.x)
    M.sub = Submodel(fixed=M.u)
    M.sub.o = Objective(expr=M.x + M.u)
    M.sub.c = Constraint(expr= M.x == M.u*M.y)

    return M
