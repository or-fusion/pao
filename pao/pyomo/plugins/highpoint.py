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
pao.pyomo.plugins.highpoint
"""

import six

from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core import Block, Objective,\
                       Var, Constraint, ComponentUID, \
                       TransformationFactory, Reference, SimpleVar
from pao.pyomo.components import SubModel
from .transform import BaseBilevelTransformation
import logging
from pyomo.core.kernel.component_map import ComponentMap

logger = logging.getLogger(__name__)


def create_submodel_hp_block(instance):
    """
    Creates highpoint relaxation with the given specified model; does
    not include any submodel or block that is deactivated
    """
    block = Block(concrete=True)

    # create a component map to keep track of id() for the
    # original object and the referenced object
    block._map = ComponentMap()

    # get the objective for the master problem
    for c in instance.component_objects(Objective, descend_into=False):
        ref = Reference(c)
        block.add_component(c.name, ref)
        block._map[c] = ref[None]

    # get the variables of the model (if there are more submodels, then
    # extraneous variables may be added to the block)
    for c in instance.component_objects(Var, sort=True, descend_into=True, active=True):
        if c.is_indexed():
            ref = Reference(c[...])
            block.add_component(c.name, ref)
            block._map[c] = ref
        else:
            ref = Reference(c)
            block.add_component(c.name, ref)
            block._map[c] = ref[None]

    # get the constraints from the main model
    for c in instance.component_objects(Constraint, sort=True, descend_into=True, active=True):
        if c.is_indexed():
            ref = Reference(c[...])
            block.add_component(c.name, ref)
            block._map[c] = ref
        else:
            ref = Reference(c)
            block.add_component(c.name, ref)
            block._map[c] = ref[None]

    # deactivate the highpoint relaxation
    # block.deactivate()

    return block


@TransformationFactory.register('pao.pyomo.highpoint',
                                doc="Generate a highpoint relaxation of the model")
class LinearHighpointTransformation(BaseBilevelTransformation):
    """
    This transformation creates a block using a SubModel object,
    which contains objective and constraint set of upper-level, and constraint set of lower-level for
    lower-level feasibility.
    """

    def _apply_to(self, model, **kwds):
        submodel_name = kwds.pop('submodel_name', 'hpr')
        self._preprocess('pao.pyomo.highpoint', model)

        map = ComponentMap()
        for key, sub in self.submodel.items():
            model.reclassify_component_type(sub, Block)

        setattr(model, submodel_name,
                    create_submodel_hp_block(model))

        model._transformation_data['pao.pyomo.highpoint'].submodel_cuid = \
            ComponentUID(model)
        model._transformation_data['pao.pyomo.highpoint'].block_cuid = \
            ComponentUID(getattr(model, submodel_name))

        for key, sub in self.submodel.items():
            model.reclassify_component_type(sub, SubModel)
