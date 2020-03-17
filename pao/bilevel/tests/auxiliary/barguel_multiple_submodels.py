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
    M._set = Set(initialize=['1','2'])
    M.x = Var(M._set)
    M.y = Var(M._set)
    M.x['1'].setlb(0)
    M.x['1'].setub(1)
    M.y['1'].setlb(0)
    M.y['1'].setub(1)
    M.x['2'].setlb(0)
    M.x['2'].setub(1)
    M.y['2'].setlb(0)
    M.y['2'].setub(1)


    # Note that the upper and lower level objectives
    # are the same. I would not actually use M.o, but
    # I am creating it here for transparency.
    M.o = Objective(expr=M.x + M.u)

    M.sub1 = SubModel(fixed=M.u)
    M.sub1.o = Objective(expr=M.x['1'] + M.u, sense=Maximize)
    M.sub1.c = Constraint(expr= M.x['1'] == M.u*M.y['1'])

    M.sub2 = SubModel(fixed=M.u)
    M.sub2.o = Objective(expr=M.x['2'] + M.u, sense=Maximize)
    M.sub2.c = Constraint(expr= M.x['2'] == M.u*M.y['2'])

    return M
