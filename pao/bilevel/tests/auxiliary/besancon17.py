#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Example from page 17
#
# Near-Optimal Robust Bilevel Optimization
#   M. Besancon, M. F. Anjos and L. Brotcorne
#   arXiv:1908.04040v5 (2019)
from pyomo.environ import *
from pao.bilevel import *

def pyomo_create_model():

    M = ConcreteModel()
    M.x = Var(within=NonNegativeReals, bounds=(0, 10000))
    M.v = Var(within=NonNegativeReals, bounds=(0, 10000))
    M.c1 = Constraint(expr=-M.x + 4 * M.v <= 11)
    M.c2 = Constraint(expr=M.x + 2 * M.v <= 13)
    M.o = Objective(expr=M.x - 10 * M.v)

    # Create a submodel
    # The argument indicates the lower-level decision variables
    M.sub = SubModel(fixed=(M.x))
    M.sub.o = Objective(expr=M.v, sense=minimize)
    M.sub.c3 = Constraint(expr=-2 * M.x - M.v <= -5)
    M.sub.c4 = Constraint(expr=5 * M.x - 4 * M.v <= 30)

    return M
# unique optimal solution (x,v) = (1,3)

if __name__ == "__main__":
    from pao.bilevel.solvers.solver2 import BilevelSolver2

    M = pyomo_create_model()
    opt = BilevelSolver2()
    opt.options.solver = "gurobi"
    opt.solve(M, tee=True)
    M.x.pprint()
    M.v.pprint()


