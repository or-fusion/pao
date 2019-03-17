#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyutilib.misc import Bunch
from pyomo.core.base import  Constraint, Objective, maximize, minimize
from pyomo.repn.standard_repn import generate_standard_repn


#
# Process linear terms from a block and return a representation of the dual.  This
# function does not change the block, but instead returns the following information:
#
# A:        The dual matrix
# b_coef:   The coefficients of the dual objective
# c_rhs:    The dual constraint right-hand side
# c_sense:  The sense of each constraint in the dual
# d_sense:  The sense of the dual objective
# vnames:   Names of the dual constraints
# cnames:   Names of the dual variables
# v_domain: A dictionary that indicates the domain of the dual variable
#               (-1: Nonpositive, 0: Unbounded, 1: Nonnegative)
#
# Note that variable bounds are treated as constraints unless they are zero.  For example,
# a variable X with bounds (-1, 1) is treated as bounded with two additional inequality
# constraints for the lower and upper bounds.  However, a variable Y with bounds (0, 3)
# is treated as a nonnegative variable with an additional constraint that defines its
# upper bound.
#
def collect_dual_representation(block, fixed):
    fixed_vars = set([id(v) for v in fixed])
    #
    # Variables are constraints of block
    # Constraints are unfixed variables of block and the parent model.
    #
    vnames = set()
    for obj in block.component_objects(Constraint, active=True):
        vnames.add((obj.getname(fully_qualified=True, relative_to=block), obj.is_indexed()))
    cnames = set()
    #for obj in block.component_objects(Var, active=True):
    #    cnames.add((obj.getname(fully_qualified=True, relative_to=block), obj.is_indexed()))

    all_vars = {}

    #
    A = {}
    b_coef = {}
    c_rhs = {}
    c_sense = {}
    d_sense = None
    v_domain = {}
    #
    # Collect objective
    #
    # TODO: Create an error if there is more than one objective
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
                    varname = var.parent_component().getname(fully_qualified=True, relative_to=block)
                except:
                    # The variable is somewhere else in the model
                    varname = var.parent_component().getname(fully_qualified=True, relative_to=block.model())
                varndx = var.index()
                all_vars[varname,varndx] = var
                c_rhs[ varname,varndx ] = coef
            nobj += 1
    if nobj == 0:
        raise RuntimeError("Error dualizing block.  No objective expression.")
    if nobj > 1:
        raise RuntimeError("Error dualizing block.  Multiple objective expressions.")
    if len(c_rhs) == 0:
        raise RuntimeError("Error dualizing block.  Objective is constant.")
    #
    # Collect constraints
    #
    for data in block.component_objects(Constraint, active=True):
        name = data.getname(relative_to=block)
        for ndx in data:
            con = data[ndx]
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
                    varname = var.parent_component().getname(fully_qualified=True, relative_to=block)
                except:
                    # The variable is somewhere else in the model
                    varname = var.parent_component().getname(fully_qualified=True, relative_to=block.model())
                varndx = var.index()
                all_vars[varname,varndx] = var
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=coef, var=name, ndx=ndx) )
            if nvars == 0:
                #
                # If a constraint has a fixed body, then don't collect it.
                #
                continue
            #
            lower_terms = generate_standard_repn(con.lower, compute_values=False) if not con.lower is None else None
            upper_terms = generate_standard_repn(con.upper, compute_values=False) if not con.upper is None else None
            #
            # Omitting these for now.  It looks like Pyomo ensures that the lower or upper bounds are constant.
            #
            ##if not lower_terms is None and not lower_terms.is_constant():
            ##    raise(RuntimeError, "Error during dualization:  Constraint '%s' has a lower bound that is non-constant")
            ##if not upper_terms is None and not upper_terms.is_constant():
            ##    raise(RuntimeError, "Error during dualization:  Constraint '%s' has an upper bound that is non-constant")
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
                    b_coef[name,ndx] = upper_terms.constant - body_terms.constant
                elif upper_terms is None:
                    #
                    # lower <= body
                    #
                    v_domain[name, ndx] = 1
                    b_coef[name,ndx] = lower_terms.constant - body_terms.constant
                else:
                    #
                    # lower <= body <= upper
                    #
                    # Dual for lower bound
                    #
                    name_ = name + '_lb_'
                    v_domain[name_, ndx] = 1
                    b_coef[name_,ndx] = lower_terms.constant - body_terms.constant
                    #
                    # Dual for upper bound
                    #
                    name_ = name + '_ub_'
                    v_domain[name_, ndx] = -1
                    b_coef[name_,ndx] = upper_terms.constant - body_terms.constant
            else:
                #
                # Equality constraint
                #
                v_domain[name, ndx] = 0
                b_coef[name,ndx] = lower_terms.constant - body_terms.constant
    #
    # Collect bound constraints
    #
    for name, ndx in all_vars:
        var = all_vars[name,ndx]
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
            c_sense[name,ndx] = 'e'
        elif bounds[0] is None:
            if bounds[1] == 0.0:
                c_sense[name,ndx] = 'g'
            else:
                c_sense[name,ndx] = 'e'
                #
                # Add constraint that defines the upper bound
                #
                name_ = name + "_upper_"
                #varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
                #varndx = data[ndx].index()
                A.setdefault(name, {}).setdefault(ndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = -1
                b_coef[name_,ndx] = bounds[1]
        elif bounds[1] is None:
            if bounds[0] == 0.0:
                c_sense[name,ndx] = 'l'
            else:
                c_sense[name,ndx] = 'e'
                #
                # Add constraint that defines the lower bound
                #
                name_ = name + "_lower_"
                #varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
                #varndx = data[ndx].index()
                A.setdefault(name, {}).setdefault(ndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = 1
                b_coef[name_,ndx] = bounds[0]
        else:
            # Bounded above and below
            c_sense[name,ndx] = 'e'
            #
            # Add constraint that defines the upper bound
            #
            name_ = name + "_upper_"
            #varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
            #varndx = data[ndx].index()
            A.setdefault(name, {}).setdefault(ndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
            #
            v_domain[name_,ndx] = -1
            b_coef[name_,ndx] = bounds[1]
            #
            # Add constraint that defines the lower bound
            #
            name_ = name + "_lower_"
            #varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
            #varndx = data[ndx].index()
            A.setdefault(name, {}).setdefault(ndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
            #
            v_domain[name_,ndx] = 1
            b_coef[name_,ndx] = bounds[0]
    #
    return (A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain)
