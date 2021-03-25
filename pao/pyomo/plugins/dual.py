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
pao.pyomo.plugins.dual
"""

from pyomo.core import Objective, Block, Var, Set, Constraint
from pyomo.core.base.block import _BlockData
from pyomo.core import TransformationFactory
from .transform import BaseBilevelTransformation
import logging

logger = logging.getLogger(__name__)

@TransformationFactory.register('pao.pyomo.linear_dual', doc="Dualize a SubModel block")
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
        subproblem_objective_weights = kwds.pop('subproblem_objective_weights', None)
        #
        # Process options
        #
        self._preprocess('pao.pyomo.linear_dual', model)
        self._fix_all()

        _dual_obj = 0.
        _dual_sense = None
        _primal_obj = 0.
        for key, sub in self.submodel.items():
            _parent = sub.parent_block()
            #
            # Generate the dual block
            #
            transform = TransformationFactory('pao.duality.linear_dual')
            dual = transform.create_using(sub, fixed=self._fixed_vardata[sub.name])
            #
            # Figure out which objective is being used
            #
            for odata in dual.component_objects(Objective, active=True):
                if subproblem_objective_weights:
                    _dual_obj += subproblem_objective_weights[key] * odata.expr
                else:
                    _dual_obj += odata.expr
                _dual_sense = odata.sense  # TODO: currently assumes all subproblems have same sense
                odata.deactivate()

            if use_dual_objective:
                #
                # Deactivate the upper-level objective, and
                # defaults to use the aggregate objective of the SubModels.
                #
                for odata in model.component_objects(Objective, active=True):
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
                for odata in sub.component_objects(Objective, active=True):
                    if subproblem_objective_weights:
                        _primal_obj += subproblem_objective_weights[key]*odata.expr
                    else:
                        _primal_obj += odata.expr

            #
            # Add the dual block
            #

            # first check if the submodel exists on a _BlockData,
            # otherwise add the dual block to the model directly
            _dual_name = key +'_dual'
            if type(_parent) == _BlockData:
                _dual_name = _dual_name.replace(_parent.name+".","")
                _parent.add_component(_dual_name, dual)
                _parent.reclassify_component_type(_dual_name, Block)
            else:
                model.add_component(_dual_name, dual)
                model.reclassify_component_type(_dual_name, Block)

            #
            # Unfix the upper variables
            #
            self._unfix_all()
            #
            # Disable the original submodel
            #
            sub.deactivate()
            for data in sub.component_map(active=True).values():
                if not isinstance(data, Var) and not isinstance(data, Set):
                    data.deactivate()

        # TODO: with multiple sub-problems, put the _obj or equiv_objs on a separate block
        if use_dual_objective:
            dual._obj = Objective(expr=_dual_obj, sense=_dual_sense)
        else:
            dual.equiv_objs = Constraint(expr=_dual_obj == _primal_obj)
