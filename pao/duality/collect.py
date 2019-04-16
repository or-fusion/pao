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
pao.duality.collect

This module defines the collect_dual_representation() function, which
collects information that is used to compute a linear dual of a given
block.
"""

__all__ = ("collect_dual_representation",)

#pylint: disable-msg=invalid-name
#pylint: disable-msg=too-many-locals
#pylint: disable-msg=too-many-branches
#pylint: disable-msg=too-many-statements

from pyutilib.misc import Bunch
from pyomo.core import  Constraint, Objective, maximize, minimize
from pyomo.repn import generate_standard_repn


def collect_dual_representation(block, fixed):
    """
    Process linear terms from a block and return information that is
    used to define the dual. This function does not change the block.

    Note that variable bounds are treated as constraints unless they are zero.  For example,
    a variable X with bounds (-1, 1) is treated as bounded with two additional inequality
    constraints for the lower and upper bounds.  However, a variable Y with bounds (0, 3)
    is treated as a nonnegative variable with an additional constraint that defines its
    upper bound.


    Arguments:
        block: The SubModel object that is dualized
        fixed: An iterable object with VarData values that are fixed in the sub-model.

    Returns: Tuple with the following values:
        A:        The dual matrix
        b_coef:   The coefficients of the dual objective
        c_rhs:    The dual constraint right-hand side
        c_sense:  The sense of each constraint in the dual
        d_sense:  The sense of the dual objective
        v_domain: A dictionary that indicates the domain of the dual variable
                      (-1: Nonpositive, 0: Unbounded, 1: Nonnegative)
    """
    #
    # Variables are constraints of block
    # Constraints are unfixed variables of block and the parent model.
    #
    fixed_vars = {id(v) for v in fixed}
    all_vars = {}

    A = {}
    b_coef = {}
    c_rhs = {}
    c_sense = {}
    d_sense = None
    v_domain = {}
    #
    # Collect objective
    #
    nobj = 0
    for odata in block.component_objects(Objective, active=True):
        for ndx in odata:
            o_terms = generate_standard_repn(odata[ndx].expr, compute_values=False)
            if odata[ndx].sense == maximize:
                d_sense = minimize
            else:
                d_sense = maximize
            for var, coef in zip(o_terms.linear_vars, o_terms.linear_coefs):
                if id(var) in fixed_vars:
                    continue
                try:
                    # The variable is in the subproblem
                    varname = var.parent_component().getname(fully_qualified=True,
                                                             relative_to=block)
                except RuntimeError:
                    # The variable is somewhere else in the model
                    varname = var.parent_component().getname(fully_qualified=True,
                                                             relative_to=block.model())
                varndx = var.index()
                all_vars[varname, varndx] = var
                c_rhs[varname, varndx] = coef
            nobj += 1
    if nobj == 0:
        raise RuntimeError("Error dualizing block.  No objective expression.")
    if nobj > 1:
        raise RuntimeError("Error dualizing block.  Multiple objective expressions.")
    if not c_rhs:
        # If len(c_rhs) == 0, then the objective is constant
        raise RuntimeError("Error dualizing block.  Objective is constant.")
    #
    # Collect constraints
    #
    for data in block.component_objects(Constraint, active=True):
        name = data.getname(relative_to=block)
        for ndx in data:
            con = data[ndx]
            if not (con.equality or con.lower is None or con.upper is None):
                dualvars = [name+"_lb_", name+"_ub_"]
            else:
                dualvars = [name]
            body_terms = generate_standard_repn(con.body, compute_values=False)
            if body_terms.is_fixed():
                #
                # If a constraint has a fixed body, then don't collect it.
                #
                continue
            nvars = 0
            for var, coef in zip(body_terms.linear_vars, body_terms.linear_coefs):
                if id(var) in fixed_vars:
                    body_terms.constant += coef*var
                    continue
                nvars += 1
                try:
                    # The variable is in the subproblem
                    varname = var.parent_component().getname(fully_qualified=True,
                                                             relative_to=block)
                except RuntimeError:
                    # The variable is somewhere else in the model
                    varname = var.parent_component().getname(fully_qualified=True,
                                                             relative_to=block.model())
                varndx = var.index()
                all_vars[varname, varndx] = var
                for dvar in dualvars:
                    A.setdefault(varname, {}).setdefault(varndx, []).append(
                                                Bunch(coef=coef, var=dvar, ndx=ndx))
            if nvars == 0:
                #
                # If a constraint has a fixed body, then don't collect it.
                #
                continue
            #
            lower_terms = generate_standard_repn(con.lower, compute_values=False) \
                                                    if not con.lower is None else None
            upper_terms = generate_standard_repn(con.upper, compute_values=False) \
                                                    if not con.upper is None else None
            #
            # It looks like Pyomo ensures that the lower or upper bounds are constant.
            # No additional checks are done.
            #
            assert(lower_terms is None or lower_terms.is_constant())
            assert(upper_terms is None or upper_terms.is_constant())
            #
            if not con.equality:
                #
                # Inequality constraint
                #
                if lower_terms is None:
                    #
                    # body <= upper
                    #
                    v_domain[name, ndx] = -1
                    b_coef[name, ndx] = upper_terms.constant - body_terms.constant
                elif upper_terms is None:
                    #
                    # lower <= body
                    #
                    v_domain[name, ndx] = 1
                    b_coef[name, ndx] = lower_terms.constant - body_terms.constant
                else:
                    #
                    # lower <= body <= upper
                    #
                    # Dual for lower bound
                    #
                    name_ = name + '_lb_'
                    v_domain[name_, ndx] = 1
                    b_coef[name_, ndx] = lower_terms.constant - body_terms.constant
                    #
                    # Dual for upper bound
                    #
                    name_ = name + '_ub_'
                    v_domain[name_, ndx] = -1
                    b_coef[name_, ndx] = upper_terms.constant - body_terms.constant
            else:
                #
                # Equality constraint
                #
                v_domain[name, ndx] = 0
                b_coef[name, ndx] = lower_terms.constant - body_terms.constant
    #
    # Collect bound constraints
    #
    for name, ndx in all_vars:
        var = all_vars[name, ndx]
        #
        # Skip fixed variables
        #
        # NOTE: This shouldn't happen because of the way we collect terms
        ##if var.fixed:
        ##    continue
        #
        # Iterate over all variable indices
        #
        bounds = var.bounds
        if bounds[0] is None and bounds[1] is None:
            c_sense[name, ndx] = 'e'
        elif bounds[0] is None:
            if bounds[1] == 0.0:
                c_sense[name, ndx] = 'g'
            else:
                c_sense[name, ndx] = 'e'
                #
                # Add constraint that defines the upper bound
                #
                name_ = name + "_upper_"
                A.setdefault(name, {}).setdefault(ndx, []).append(
                    Bunch(coef=1.0, var=name_, ndx=ndx))
                #
                v_domain[name_, ndx] = -1
                b_coef[name_, ndx] = bounds[1]
        elif bounds[1] is None:
            if bounds[0] == 0.0:
                c_sense[name, ndx] = 'l'
            else:
                c_sense[name, ndx] = 'e'
                #
                # Add constraint that defines the lower bound
                #
                name_ = name + "_lower_"
                A.setdefault(name, {}).setdefault(ndx, []).append(
                    Bunch(coef=1.0, var=name_, ndx=ndx))
                #
                v_domain[name_, ndx] = 1
                b_coef[name_, ndx] = bounds[0]
        else:
            # Bounded above and below
            c_sense[name, ndx] = 'e'
            #
            # Add constraint that defines the upper bound
            #
            name_ = name + "_upper_"
            A.setdefault(name, {}).setdefault(ndx, []).append(Bunch(coef=1.0, var=name_, ndx=ndx))
            #
            v_domain[name_, ndx] = -1
            b_coef[name_, ndx] = bounds[1]
            #
            # Add constraint that defines the lower bound
            #
            name_ = name + "_lower_"
            #varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
            #varndx = data[ndx].index()
            A.setdefault(name, {}).setdefault(ndx, []).append(Bunch(coef=1.0, var=name_, ndx=ndx))
            #
            v_domain[name_, ndx] = 1
            b_coef[name_, ndx] = bounds[0]
    #
    return (A, b_coef, c_rhs, c_sense, d_sense, v_domain)
