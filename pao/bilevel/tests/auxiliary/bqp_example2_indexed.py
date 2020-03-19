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

    M.x_set = Set(initialize=['1'])
    M.x = Var(M.x_set, within=Binary, initialize=0)

    M.u_set = Set(initialize=['1'])
    M.u = Var(M.x_set, initialize=0)
    M.x['1'].setlb(0)

    M.y_set = Set(initialize=['1','2','3','4'])
    M.y = Var(M.y_set)
    M.y['1'].setlb(1)
    M.y['2'].setlb(-100)
    M.y['2'].setub(2)
    M.y['4'].setlb(3)
    M.y['4'].setub(4)

    M.o = Objective(expr=M.u['1'] - 4 * M.y['1'])

    M.sub = SubModel(fixed=(M.u, M.x))
    M.sub.o = Objective(expr=-11 * M.x['1'] - 12 * M.x['1'] * M.y['1'] - M.y['2'] - 9 * M.y['3'], sense=maximize)
    M.sub.c1 = Constraint(expr=M.u['1'] + 13 * M.x['1'] * M.y['1'] + 5 * M.y['1'] <= 19)
    M.sub.c2 = Constraint(expr=20 <= 2 * M.u['1'] + 6 * M.y['1'] + 14 * M.x['1'] * M.y['2'] + 10 * M.y['3'])
    M.sub.c3 = Constraint(expr=32 == 4 * M.u['1'] + 8 * M.y['1'] + 15 * M.x['1'] * M.y['4'])
    M.sub.c4 = Constraint(expr=inequality(22, 3 * M.u['1'] + 7 * M.y['1'] + 16 * M.x['1'] * M.y['3'], 28))

    return M

if __name__ == "__main__":
    pyomo_create_model()