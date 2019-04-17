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
pao.bilevel.plugins.dual
"""

from pyomo.core import Objective, Block, Var, Set
from pyomo.core import TransformationFactory
from .transform import BaseBilevelTransformation


@TransformationFactory.register('pao.bilevel.linear_dual', doc="Dualize a SubModel block")
class LinearDualBilevelTransformation(BaseBilevelTransformation):
    """
    This transformation creates a linear dualization of a sub-model.
    """

    def _apply_to(self, model, **kwds):
        submodel_name = kwds.pop('submodel', None)
        #
        # Process options
        #
        submodel = self._preprocess('pao.bilevel.linear_dual', model, sub=submodel_name)
        self._fix_all()
        #
        # Generate the dual
        #
        transform = TransformationFactory('pao.duality.linear_dual')
        dual = transform.create_using(submodel, fixed=self._fixed_vardata)
        setattr(model, self._submodel+'_dual', dual)
        model.reclassify_component_type(self._submodel+'_dual', Block)
        #
        # Deactivate the original subproblem and upper-level objective
        #
        for odata in submodel._parent().component_map(Objective, active=True).values():
            odata.deactivate()
        submodel.deactivate()
        #
        # Unfix the upper variables
        #
        self._unfix_all()
        #
        # Disable the original submodel
        #
        sub = getattr(model, self._submodel)
        for data in sub.component_map(active=True).values():
            if not isinstance(data, Var) and not isinstance(data, Set):
                data.deactivate()
