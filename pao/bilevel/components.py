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
pao.bilevel.components

This module defines Pyomo components used to declare bilevel programs.
"""

__all__ = ("SubModel",)

#pylint: disable-msg=too-many-ancestors

from pyomo.core import SimpleBlock, ModelComponentFactory, Component


@ModelComponentFactory.register("A submodel in a bilevel program")
class SubModel(SimpleBlock):
    """
    This model component defines a sub-model in a bilevel
    program.
    """

    def __init__(self, *args, **kwargs):
        """Constructor"""
        #
        # Collect kwargs for SubModel
        #
        _rule = kwargs.pop('rule', None)
        _fixed = kwargs.pop('fixed', None)
        #_var = kwargs.pop('var', None)
        #
        # Initialize the SimpleBlock
        #
        kwargs.setdefault('ctype', SubModel)
        SimpleBlock.__init__(self, *args, **kwargs)
        #
        # Initialize from kwargs
        #
        self._rule = _rule
        if isinstance(_fixed, Component):
            self._fixed = [_fixed]
        else:
            self._fixed = _fixed
        #if isinstance(_var, Component):
        #    self._var = [_var]
        #else:
        #    self._var = _var
