#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from six import iteritems

import pyomo.common
from pyomo.core.base import (Transformation,
                             TransformationFactory,
                             Var,
                             Constraint,
                             Objective,
                             Set,
                             minimize,
                             NonNegativeReals,
                             NonPositiveReals,
                             Reals,
                             Block,
                             Model,
                             ConcreteModel)
from pao.duality.collect import collect_dual_representation

def load(): #pragma:nocover
    pass

logger = logging.getLogger('pao')


#
# Generate the dual of a block
#
# Creates a new block that is the dual of the block object.  The resulting
# block contains variables and constraints whose names are the dual names of
# the primal block.  Note that this involves a many string operations.  A quicker
# operations could be executed, but it would generate a dual representation that is
# difficult to interpret.
#
# Note that the dualization of a maximization problem is performed by negating objective
# and right-hand side coefficients after dualizing the corresponding minimization problem. 
# This suggestion is made by Dimitri Bertsimas and John Tsitsiklis in section 4.2 page 143 of 
# "Introduction to Linear Optimization"
#
# If the block is a model object, then this returns a ConcreteModel.  Otherwise, it
# returns a Block.
#
def create_linear_dual(block, fixed):
    #
    # Collect linear terms from the block
    #
    A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain = collect_dual_representation(block, fixed)
    #
    # Construct the block
    #
    if isinstance(block, Model):
        dual = ConcreteModel()
    else:
        dual = Block()
    dual.construct()
    _vars = {}

    # Return variable object from name and index (if applicable)
    def getvar(name, ndx=None):
        v = _vars.get((name,ndx), None)
        if v is None:
            v = Var()
            if ndx is None:
                v_name = name
            elif type(ndx) is tuple:
                v_name = "%s[%s]" % (name, ','.join(map(str,ndx)))
            else:
                v_name = "%s[%s]" % (name, str(ndx))
            setattr(dual, v_name, v)
            _vars[name,ndx] = v
        return v
    #
    # Construct the objective
    # Note that dualization of a maximization problem is handled by simply negating the objective and 
    # left-hand side coefficients while keeping the dual sense.
    #
    if d_sense == minimize:
        dual.o = Objective(expr=sum(-   b_coef[name,ndx]*getvar(name,ndx) for name,ndx in b_coef), sense=d_sense)
        rhs_multiplier = -1
    else:
        dual.o = Objective(expr=sum(b_coef[name,ndx]*getvar(name,ndx) for name,ndx in b_coef), sense=d_sense)
        rhs_multiplier = 1
    #
    # Construct the constraints from dual A matrix
    #
    for cname in A:
        for ndx, terms in iteritems(A[cname]):

            # Build left-hand side of constraint
            expr = 0
            for term in terms:
                expr += term.coef * getvar(term.var, term.ndx)

            #
            # Assign right-hand side coefficient
            # Note that rhs_multiplier is 1 if the dual is a maximization problem and -1 otherwise
            #
            rhsval = rhs_multiplier*c_rhs.get((cname,ndx), 0.0)

            # Using the correct inequality or equality
            if c_sense[cname, ndx] == 'e':
                e = expr - rhsval == 0
            elif c_sense[cname, ndx] == 'l':
                e = expr - rhsval <= 0
            else:
                e = expr - rhsval >= 0
            c = Constraint(expr=e)

            # Build constraint name
            if ndx is None:
                c_name = cname
            elif type(ndx) is tuple:
                c_name = "%s[%s]" % (cname, ','.join(map(str,ndx)))
            else:
                c_name = "%s[%s]" % (cname, str(ndx))

            # Add new constraint along with its name to the dual
            setattr(dual, c_name, c)

        # Set variable domains
        for (name, ndx), domain in iteritems(v_domain):
            v = getvar(name, ndx)
            flag = type(ndx) is tuple and (ndx[-1] == 'lb' or ndx[-1] == 'ub')
            if domain == 1:
                v.domain = NonNegativeReals
            elif domain == -1:
                v.domain = NonPositiveReals
            else:
                # TODO: verify that this case is possible
                # BA: This is possible when the variable's corresponding constraint is an equality
                v.domain = Reals

    return dual


#
# This transformation creates a new block that
# is the dual of the specified block.  If no block is
# specified, then the entire model is dualized.
#
# This returns a new Block object.
#
@TransformationFactory.register('pao.duality.linear_dual', doc="Dualize a linear model")
class LinearDual_PyomoTransformation(Transformation):


    def __init__(self):
        super(LinearDual_PyomoTransformation, self).__init__()

    def _create_using(self, instance, **kwds):
        bname = kwds.get('block',None)
        fixed = kwds.get('fixed', [])
        #
        # Iterate over the model collecting variable data,
        # until the block is found.
        #
        block = None
        if bname is None:
            block = instance
        else:
            for (name, data) in instance.component_map(Block, active=True).items():
                if name == bname:
                    block = data
                    break
        if block is None:
            raise RuntimeError("Missing block: "+bname)
        #
        # Generate the dual
        #
        return create_linear_dual(block, fixed)

