#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Example 2.7 from
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)
from pyomo.environ import *
from pao.bilevel import *

def pyomo_create_model():

    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.v = Var()
    model.o = Objective(expr=model.x, sense=minimize)
    model.c1 = Constraint(expr=model.v >= 1 - model.x/10)

    # Create a submodel
    # The argument indicates the lower-level decision variables
    model.sub = SubModel(fixed=model.x)
    model.sub.o = Objective(expr=model.v, sense=maximize)
    model.sub.c1 = Constraint(expr=1 + model.x/10 >= model.v)

    return model


# unique optimal solution (x,v) = (0,1)

if __name__ == "__main__":
    pyomo_create_model()