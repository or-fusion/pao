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
pao.duality.plugins

This module defines the transformation plugin for linear dualization.
"""

#pylint: disable-msg=invalid-name
#pylint: disable-msg=too-many-locals
#pylint: disable-msg=too-many-branches

import logging

logger = logging.getLogger(__name__)

from pyomo.core import (Transformation,
                        TransformationFactory, Block)
from pao.duality.collect import create_linear_dual_from

def load(): #pragma:nocover
    """
    No operations are needed to load these plugins.
    """

logger = logging.getLogger('pao')


@TransformationFactory.register('pao.duality.linear_dual', doc="Dualize a linear model")
class LinearDual_PyomoTransformation(Transformation):
    """
    This transformation creates a new block that
    is the dual of the specified block.  If no block is
    specified, then the entire model is dualized.
    """

    def _create_using(self, model, **kwds):
        bname = kwds.pop('block', None)
        #
        # Iterate over the model collecting variable data,
        # until the block is found.
        #
        block = None
        if bname is None:
            block = model
        else:
            for (name, data) in model.component_map(Block, active=True).items():
                if name == bname:
                    block = data
                    break
        if block is None:
            raise RuntimeError("Missing block: "+bname)
        #
        # Generate the dual
        #
        return create_linear_dual_from(block, **kwds)
