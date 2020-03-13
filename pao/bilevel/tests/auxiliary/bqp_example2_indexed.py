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


def pyomo_create_model():
    M = ConcreteModel()
    M.x1 = Var(bounds=(0, None))
    M.x2 = Var(within=Binary)
    M.y_set = Set(initialize=['1','2','3','4'])
    M.y = Var(M.y_set)
    M.y['1'].setlb(1)
    M.y['2'].setlb(-100)
    M.y['2'].setub(2)
    M.y['4'].setlb(3)
    M.y['4'].setub(4)

    M.o = Objective(expr=M.x1 - 4 * M.y['1'])

    M.sub = SubModel(fixed=(M.x1, M.x2))
    M.sub.o = Objective(expr=-11 * M.x2 - 12 * M.x2 * M.y['1'] - M.y['2'] - 9 * M.y['3'], sense=maximize)
    M.sub.c1 = Constraint(expr=M.x1 + 13 * M.x2 * M.y['1'] + 5 * M.y['1'] <= 19)
    M.sub.c2 = Constraint(expr=20 <= 2 * M.x1 + 6 * M.y['1'] + 14 * M.x2 * M.y['2'] + 10 * M.y['3'])
    M.sub.c3 = Constraint(expr=32 == 4 * M.x1 + 8 * M.y['1'] + 15 * M.x2 * M.y['4'])
    M.sub.c4 = Constraint(expr=inequality(22, 3 * M.x1 + 7 * M.y['1'] + 16 * M.x2 * M.y['3'], 28))

    return M

if __name__ == "__main__":
    pyomo_create_model()