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
pao.bilevel.plugins

This package defines plugins in pao.bilevel
"""

#pylint: disable-msg=unused-import

def load():
    """
    Import the plugins defined in pao.bilevel
    """
    import pao.bilevel.plugins.dual
    import pao.bilevel.plugins.lcp
    import pao.bilevel.solvers.solver1
    import pao.bilevel.solvers.solver2
    import pao.bilevel.solvers.solver3
    import pao.bilevel.solvers.solver4

