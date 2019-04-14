"""
pao package

Defines modeling techniques and solvers for representing optimization
problems with adversarial behavior.  PAO currently supports bilevel
programming.

Importing PAO initializes the Pyomo environment and then registers
the pao plugins.  We assume that a user will never import symbols from
pao directly:

    from pao import *

Instead, users should import symbols directly from pao sub-packages:

    from pao.bilevel import *
"""

__all__ = ()

import pyomo.environ
import pao.duality.plugins
import pao.bilevel.plugins
pao.bilevel.plugins.load()
