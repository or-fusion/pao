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
pao.pyomo.components

This module defines Pyomo components used to declare multi-level problems.
"""

__all__ = ("SubModel",)

from pyomo.core import SimpleBlock, ModelComponentFactory, Component
import pyomo.core.base.component_order


@ModelComponentFactory.register("A submodel in a multi-level problem")
class SubModel(SimpleBlock):
    """
    This Pyomo model component defines a sub-model in a multi-level problem.

    Pyomo models can include nested and parallel SubModel components to express 
    complex multi-level problems.
    """

    def __init__(self, *args, **kwargs):
        """
        The constructor for SubModel components.

        Keyword Args
        ------------
        fixed
            A Pyomo variable or a list of Pyomo variables that are optimized by upper-levels in the model.

        Parameters
        ----------
        *args
            Arguments passed to the SimpleBlock base class.
        **kwargs
            Other keyword arguments passed to the SimpleBlock base class.
        """
        #
        # Collect kwargs for SubModel
        #
        _rule = kwargs.pop('rule', None)
        _fixed = kwargs.pop('fixed', None)
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

#
# Register 'SubModel' with Pyomo's core components.  This
# ensure that the model display() method includes SubModel
# components.
#
pyomo.core.base.component_order.display_name[SubModel] = 'SubModel'
pyomo.core.base.component_order.display_items.append(SubModel)
