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
#from pao.bilevel.components import SubModel
#from .transform import BaseBilevelTransformation
import logging

logger = logging.getLogger(__name__)


def create_submodel_hp_block(instance):
    """
    Creates highpoint relaxation with the given specified model; does
    not include any submodel or block that is deactivated
    """
    if not instance.active:
        raise RuntimeError("The current model is deactivated and cannot build the highpoint relaxation.")

    block = Block(concrete=True)

    # get the objective for the master problem
    for c in instance.component_objects(Objective, descend_into=False):
        block.add_component(c.name, Reference(c))

    # get the variables of the model (if there are more submodels, then
    # extraneous variables may be added to the block)
    for c in instance.component_objects(Var, descend_into=False):
        block.add_component(c.name, Reference(c))

    # get the constraints from the main model
    for c in instance.component_objects(Constraint, descend_into=False):
        block.add_component(c.name, Reference(c))

    # get the constraints from the submodel of interest
    # add anything to the hpr that is deactivated
    # allow user to pass in their HPR block
    def _process_nested_block(model):
        for _block in model.component_objects(Block, descend_into=False):
            for c in _block.component_objects(Constraint, descend_into=False):

                # modify the name of the block
                check = _block.find_component(c.name)
                if check is None:
                    block.add_component(c.name, Reference(c))
                else:
                    block.add_component(_block.name + '_' + c.name, Reference(c))
            _process_nested_block(_block)

    _process_nested_block(instance)

    # deactivate the highpoint relaxation
    block.deactivate()

    return block

#
# @TransformationFactory.register('pao.bilevel.highpoint',
#                                 doc="Generate a highpoint relaxation of the model")
# class LinearHighpointTransformation(BaseBilevelTransformation):
#     """
#     This transformation creates a block using a SubModel object,
#     which contains objective and constraint set of upper-level, and constraint set of lower-level for
#     lower-level feasibility.
#     """
#
#     def _apply_to(self, model, **kwds):
#         submodel_name = kwds.pop('submodel', None)
#
#         #
#         # Process options
#         #
#         self._preprocess('pao.bilevel.highpoint', model)
#
#         def _sub_transformation(model, sub, key):
#             model.reclassify_component_type(sub, Block)
#             #
#             # Create a block with optimality conditions
#             #
#             setattr(model, key +'_hp',
#                     create_submodel_hp_block(model, sub))
#             model._transformation_data['pao.bilevel.highpoint'].submodel_cuid =\
#                 ComponentUID(sub)
#             model._transformation_data['pao.bilevel.highpoint'].block_cuid =\
#                 ComponentUID(getattr(model, key +'_hp'))
#
#         if not submodel_name is None:
#             lookup = {value: key for key, value in self.submodel}
#             sub = getattr(model,submodel_name)
#             if sub:
#                 _sub_transformation(model, sub, lookup[sub])
#             return
#
#         for key, sub in self.submodel.items():
#             _sub_transformation(model, sub, key)
#

if __name__ == '__main__':
    from pyomo.environ import *
    from pao.bilevel import *

    M = ConcreteModel()
    M.x1 = Var(bounds=(0, None))
    M.x2 = Var(within=Binary)
    M.y1 = Var(bounds=(1, None))
    M.y2 = Var(bounds=(-100, 2))
    M.y3 = Var(bounds=(None, None))
    M.y4 = Var(bounds=(3, 4))
    M.o = Objective(expr=M.x1 - 4 * M.y1)

    M.sub = SubModel(fixed=(M.x1, M.x2))
    M.sub.o = Objective(expr=11 * M.x2 + 12 * M.x2 * M.y1 + M.y2 + 9 * M.y3)
    M.sub.c1 = Constraint(expr=M.x1 + 13 * M.x2 * M.y1 + 5 * M.y1 <= 19)
    M.sub.c2 = Constraint(expr=20 <= 2 * M.x1 + 6 * M.y1 + 14 * M.x2 * M.y2 + 10 * M.y3)
    M.sub.c3 = Constraint(expr=32 == 4 * M.x1 + 8 * M.y1 + 15 * M.x2 * M.y4)
    M.sub.c4 = Constraint(expr=inequality(22, 3 * M.x1 + 7 * M.y1 + 16 * M.x2 * M.y3, 28))

    M.T = range(2)
    M.block = Block(M.T)

    for t in M.T:
        M.block[t].c1 = Constraint(expr=M.x1 + 13 * M.x2 * M.y1 + 5 * M.y1 <= 19)

    M.hp = create_submodel_hp_block(M)
    M.hp.pprint()
