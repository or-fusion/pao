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
pao.pyomo.util

This module defines utility functions for working with Pyomo multi-level problems.
"""

from pyomo.core import Reference, Var, Transformation, Param, Set
#pylint: disable-msg=too-many-ancestors

from pyomo.core import SimpleBlock, ModelComponentFactory, Component


def varref(model, origin=None, vars=None):
    """
    This helper function enables variables to be locally referenced
    from within a given block on the model, since all variables on
    a bilevel model exist only on the parent_block() for ConcreteModel()
    """

    # default to root block
    if not origin:
        origin = model.root_block()

    # intersection of 2 Pyomo var lists
    def _intersection(list1, list2):
        list3 = list()
        for l1 in list1:
            for l2 in list2:
                if l1.name == l2.name:
                    list3.append(l1)
        return list3

    # list of variables to add references for to the nested block; if vars is specified,
    # vars must exist on the origin block
    var_list = [var for var in origin.component_objects(Var, descend_into=False)]
    if vars:
        var_list = _intersection(vars, var_list)

    # if variable name already exists on nested block, raise a RuntimeError (in Pyomo); otherwise add reference
    # for variable onto the nested block
    for _var in var_list:
        model.add_component(_var.name, Reference(_var))


def dataref(model, origin=None):
    """
    This helper function enables params and sets to be locally referenced
    from within a given block on the model, since all variables on
    a bilevel model exist only on the parent_block() for ConcreteModel()
    """
    # default to parent block
    if not origin:
        origin = model.root_block()

    for p in origin.component_objects(Param, descend_into=False):
        model.add_component(p.name, Reference(p))

    for s in origin.component_objects(Set, descend_into=False):
        model.add_component(s.name, Reference(s))


def deactivate_submodels(model):
    """
    This is a helper function to deactivate all SubModel blocks
    """
    for b in model.component_objects(SubModel):
        b.deactivate()


def activate_submodels(model):
    """
    This is a helper function to activate all SubModel blocks
    """
    for b in model.component_objects(SubModel):
        b.activate()

