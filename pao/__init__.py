#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
pao

This package defines modeling techniques and solvers for representing
optimization problems with adversarial behavior.  PAO currently supports
bilevel programming.

Importing pao initializes the Pyomo environment and then registers
the pao plugins.  We assume that a user will never import symbols from
pao directly:

    $ from pao import *

Instead, users should import symbols directly from pao sub-packages:

    $ from pao.pyomo import *

Version: %s
"""

from pao._version import __version__
__doc__ = __doc__ % __version__

__all__ = ('__version__')

import pyomo.environ
import pao.mpr
import pao.pyomo
#import pao.duality.plugins
import pao.pyomo.plugins
pao.pyomo.plugins.load()

from pao.common import Solver
