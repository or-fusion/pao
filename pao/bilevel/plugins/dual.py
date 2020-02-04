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

from pyomo.core import Objective, Block, Var, Set, Constraint
from pyomo.core import TransformationFactory
from .transform import BaseBilevelTransformation
import logging

logger = logging.getLogger(__name__)

@TransformationFactory.register('pao.bilevel.linear_dual', doc="Dualize a SubModel block")
class LinearDualBilevelTransformation(BaseBilevelTransformation):
    """
    This transformation creates a linear dualization of a sub-model.
    This is suitable for interdiction problems, where the upper and lower
    problems have opposite objectives.

    The use_dual_objective can be used to simplify the final problem 
    representation sligthly, in the case where there are no constraints in the 
    upper-level problem.
    """

    def _apply_to(self, model, **kwds):
        submodel_name = kwds.pop('submodel', None)
        use_dual_objective = kwds.pop('use_dual_objective', False)
        #
        # Process options
        #
        self._preprocess('pao.bilevel.linear_dual', model)
        self._fix_all()

        for key, sub in self.submodel.items():
            #
            # Generate the dual block
            #
            transform = TransformationFactory('pao.duality.linear_dual')
            dual = transform.create_using(sub, fixed=self._fixed_vardata)
            #
            # Figure out which objective is being used
            #
            if use_dual_objective:
                #
                # Deactivate the upper-level objective
                #
                # TODO: Warn if there are multiple objectives?
                #
                for odata in sub._parent().component_map(Objective, active=True).values():
                    odata.deactivate()
            else:
                #
                # Add a constraint that maps the dual objective to the primal objective
                #
                # NOTE: It might be numerically more stable to replace the upper
                # objective with a variable, and then set the dual equal to that variable.
                # But that transformation would not be limited to the submodel.  If that's
                # an issue for a user, they can make that change, and see the benefit.
                #
                dual_obj = None
                for odata in dual.component_objects(Objective, active=True):
                    dual_obj = odata
                    dual_obj.deactivate()
                    break
                primal_obj = None
                for odata in sub.component_objects(Objective, active=True):
                    primal_obj = odata
                    break
                dual.equiv_objs = Constraint(expr=dual_obj.expr == primal_obj.expr)
            #
            # Add the dual block
            #
            setattr(model, key +'_dual', dual)
            model.reclassify_component_type(key +'_dual', Block)
            #
            # Unfix the upper variables
            #
            self._unfix_all()
            #
            # Disable the original submodel
            #
            # Q: Are the last steps redundant?  Will we recurse into deactivated blocks?
            #
            sub.deactivate()
            for data in sub.component_map(active=True).values():
                if not isinstance(data, Var) and not isinstance(data, Set):
                    data.deactivate()
