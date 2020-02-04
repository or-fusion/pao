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


def create_submodel_hp_block(instance, submodel):
    """
    Creates highpoint relaxation with the given specified submodel
    """
    block = Block(concrete=True)

    # get the objective for the master problem
    for c in instance.component_objects(Objective, descend_into=False):
        if c.parent_block() == instance:
            block.add_component(c.name, Reference(c))

    # get the variables of the model (if there are more submodels, then
    # extraneous variables may be added to the block)
    for c in instance.component_objects(Var, descend_into=False):
        if c.parent_block() == instance:
            block.add_component(c.name, Reference(c))

    # get the constraints from the main model
    for c in instance.component_objects(Constraint, descend_into=False):
        if c.parent_block() == instance:
            block.add_component(c.name, Reference(c))

    # get the constraints from the submodel of interest
    for _block in instance.component_objects(Block, descend_into=False):
        if _block == submodel:
            for c in _block.component_objects(Constraint, descend_into=False):
                check = block.find_component(c.name) # checks to see if a constraint by the same name is on the main block
                # this is feasible due to local scoping of each block, but we need unique names for the highpoint relaxation
                if check is None:
                    block.add_component(c.name, Reference(c))
                else:
                    block.add_component(submodel.name + '_' + c.name, Reference(c))

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
        submodel_name = kwds.pop('submodel', None)

        #
        # Process options
        #
        self._preprocess('pao.bilevel.highpoint', model)

        def _sub_transformation(model, sub, key):
            model.reclassify_component_type(sub, Block)
            #
            # Create a block with optimality conditions
            #
            setattr(model, key +'_hp',
                    create_submodel_hp_block(model, sub))
            model._transformation_data['pao.bilevel.highpoint'].submodel_cuid =\
                ComponentUID(sub)
            model._transformation_data['pao.bilevel.highpoint'].block_cuid =\
                ComponentUID(getattr(model, key +'_hp'))

        if not submodel_name is None:
            lookup = {value: key for key, value in self.submodel}
            sub = getattr(model,submodel_name)
            if sub:
                _sub_transformation(model, sub, lookup[sub])
            return

        for key, sub in self.submodel.items():
            _sub_transformation(model, sub, key)

