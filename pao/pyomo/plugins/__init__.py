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
pao.pyomo.plugins

This package defines plugins in pao.pyomo
"""

#pylint: disable-msg=unused-import

def load():
    """
    Import the plugins defined in pao.pyomo
    """
    import pao.pyomo.solvers.mpr_solvers

