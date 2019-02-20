#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Constraint, Objective, Block
from pyomo.repn import generate_standard_repn
from pyomo.core.base.plugin import TransformationFactory
from pyomo.core.base import Var, Set
from pao.bilevel.plugins.transform import Base_BilevelTransformation

import logging
logger = logging.getLogger('pao')


@TransformationFactory.register('pao.bilevel.linear_dual', doc="Dualize a SubModel block")
class LinearDual_BilevelTransformation(Base_BilevelTransformation):

    def __init__(self):
        super(LinearDual_BilevelTransformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        #
        # Process options
        #
        submodel = self._preprocess('pao.bilevel.linear_dual', instance, **kwds)
        self._fix_all()
        #
        # Generate the dual
        #
        setattr(instance, self._submodel+'_dual', self._dualize(submodel, self._unfixed_upper_vars))
        instance.reclassify_component_type(self._submodel+'_dual', Block)
        #
        # Deactivate the original subproblem and upper-level objective
        #
        for (oname, odata) in submodel._parent().component_map(Objective, active=True).items():
            odata.deactivate()
        submodel.deactivate()
        #
        # Unfix the upper variables
        #
        self._unfix_all()
        #
        # Disable the original submodel
        #
        sub = getattr(instance,self._submodel)
        # TODO: Cache the list of components that were deactivated
        for (name, data) in sub.component_map(active=True).items():
            if not isinstance(data,Var) and not isinstance(data, Set):
                data.deactivate()


    def _dualize(self, submodel, unfixed):
        """
        Generate the dual of a submodel
        """ 
        transform = TransformationFactory('pao.duality.linear_dual')
        return transform._dualize(submodel, unfixed)

