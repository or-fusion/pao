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

#import os
#from os.path import abspath, dirname, normpath, join
#currdir = dirname(abspath(__file__))
#exdir = currdir

import pyutilib.th as unittest
from pyomo.environ import *
import pao


class Test(unittest.TestCase):

    def test_missing_model(self):
        m = ConcreteModel()
        m.x = Var()
        xfrm = TransformationFactory('pao.bilevel.linear_dual')
        self.assertRaises(RuntimeError, xfrm.apply_to, m)



if __name__ == "__main__":
    unittest.main()
