#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Test transformations for linear duality
#

import pyutilib.th as unittest
from pyomo.environ import *
import pao


class Test(unittest.TestCase):

    def test_missing_submodel(self):
        m = ConcreteModel()
        m.x = Var()
        xfrm = TransformationFactory('pao.bilevel.linear_dual')
        self.assertRaises(RuntimeError, xfrm.apply_to, m)

    def test_missing_fixed_or_unfixed(self):
        m = ConcreteModel()
        m.x = Var()
        m.o = Objective(expr=m.x)
        m.sub = pao.bilevel.SubModel()
        m.sub.o = Objective(expr=-m.x)
        xfrm = TransformationFactory('pao.bilevel.linear_dual')
        self.assertRaises(RuntimeError, xfrm.apply_to, m)


if __name__ == "__main__":
    unittest.main()
