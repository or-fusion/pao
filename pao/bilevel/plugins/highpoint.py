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
pao.bilevel.plugins.highpoint
"""

import six

from pyomo.core import Block, VarList, ConstraintList, Objective,\
                       Var, Constraint, maximize, ComponentUID, Set,\
                       TransformationFactory
from pyomo.repn import generate_standard_repn
from pyomo.mpec import ComplementarityList, complements
from .transform import BaseBilevelTransformation
import logging

logger = logging.getLogger(__name__)

def create_submodel_hp_block(): pass

@TransformationFactory.register('pao.bilevel.highpoint',
                                doc="Generate a highpoint representation of the model")
class LinearComplementarityBilevelTransformation(BaseBilevelTransformation):
    """
    This transformation creates a block using a SubModel object,
    which contains objective and constraint set of upper-level, and constraint set of lower-level for
    lower-level feasibility.
    """

    def _apply_to(self, model, **kwds):
        deterministic = kwds.pop('deterministic', False)
        submodel_name = kwds.pop('submodel', None)
        #
        # Process options
        #
        self._preprocess('pao.bilevel.highpoint', model)
        for (key1,key2), sub in self.submodel.items():
            model.reclassify_component_type(sub, Block)

            #
            # Create a block with highpoint formulation
            #
            setattr(model, str(key1)+'_hp',
                    create_submodel_hp_block(model, sub, deterministic,
                                              self.fixed_vardata[(key1,key2)]))
            model._transformation_data['pao.bilevel.highpoint'].submodel_cuid =\
                ComponentUID(sub)
            model._transformation_data['pao.bilevel.highpoint'].block_cuid =\
                ComponentUID(getattr(model, str(key1)+'_hp'))
            #
            # Disable the original submodel and
            #
            for data in sub.component_map(active=True).values():
                if not isinstance(data, Var) and not isinstance(data, Set):
                    data.deactivate()

