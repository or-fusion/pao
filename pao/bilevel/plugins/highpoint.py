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
                       TransformationFactory, Model, Reference
from pao.bilevel.components import SubModel
from .transform import BaseBilevelTransformation
import logging

logger = logging.getLogger(__name__)


def create_submodel_hp_block(instance):
    """
    Creates highpoint relaxation with the given specified model; does
    not include any submodel or block that is deactivated
    """
    block = Block(concrete=True)

    # get the objective for the master problem
    for c in instance.component_objects(Objective, descend_into=False):
        block.add_component(c.name, Reference(c))

    # get the variables of the model (if there are more submodels, then
    # extraneous variables may be added to the block)
    for c in instance.component_objects(Var, sort=True, descend_into=True, active=True):
        block.add_component(c.name, Reference(c))

    # get the constraints from the main model
    for c in instance.component_objects(Constraint, sort=True, descend_into=True, active=True):
        block.add_component(c.name, Reference(c))

    # deactivate the highpoint relaxation
    block.deactivate()

    return block


@TransformationFactory.register('pao.bilevel.highpoint',
                                doc="Generate a highpoint relaxation of the model")
class LinearHighpointTransformation(BaseBilevelTransformation):
    """
    This transformation creates a block using a SubModel object,
    which contains objective and constraint set of upper-level, and constraint set of lower-level for
    lower-level feasibility.
    """

    def _apply_to(self, model, **kwds):
        self._preprocess('pao.bilevel.highpoint', model)

        for key, sub in self.submodel.items():
            model.reclassify_component_type(sub, Block)

        setattr(model, 'hpr',
                    create_submodel_hp_block(model))

        model._transformation_data['pao.bilevel.highpoint'].submodel_cuid = \
            ComponentUID(model)
        model._transformation_data['pao.bilevel.highpoint'].block_cuid = \
            ComponentUID(getattr(model, 'hpr'))
