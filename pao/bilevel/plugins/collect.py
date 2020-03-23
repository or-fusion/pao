# coding=utf-8
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
pao.bilevel.plugins

This module defines the collect_bilevel_matrix_representation() function, which
collects information that is used to in bilevel algorithm development.
"""

from pyutilib.misc import Bunch
from pyomo.repn import generate_standard_repn
from pyomo.core import (Var,
                        Constraint,
                        Objective,
                        minimize,
                        maximize,
                        NonNegativeReals,
                        NonPositiveReals,
                        Reals,
                        Block,
                        Model,
                        ConcreteModel)


def collect_bilevel_matrix_representation(block, fixed, unfixed):
    """
    Collects the terms for each ==, >= and <= constraints
    for the given local variable scope, per variable type (c:continuous,
    b:binary, i:integer). Nested Blocks (or SubModels) are not considered
    within local scope.

    The terms will be denoted as follows:
    A,          varref,         sense,          b_coef
    A^{c} *     x^{c}           {==, >=, <=}    b^{c}
    A^{b} *     x^{b}           {==, >=, <=}    b^{b}
    A^{i} *     x^{i}           {==, >=, <=}    b^{i}

    Arguments:
        block: A Pyomo Model, Block, or SubModel that will be processed to
                collect terms for Ax <= b, Ax >= b, and Ax == b.
        fixed: An iterable object with Variable and VarData values that are fixed in
                this model.  All other variables are assumed to be unfixed.
        unfixed: An iterable object with Variable and VarData values that are not fixed
                in this model.  All other variables are assumed to be fixed.

    Returns: Tuple with the following values:
    A^{c}:          The A coefficient matrix associate to continuous vars
    x^{c}:          The continuous vars
    sense^{c}:      The sense of the constraints to the continuous vars, per constraint
    b_coef^{c}:     The coefficients of the right-hand side for the continuous vars
    A^{b}:          The A coefficient matrix associate to binary vars
    x^{b}:          The binary vars
    sense^{b}:      The sense of the constraints to the binary vars, per constraint
    b_coef^{b}:     The coefficients of the right-hand side for the binary vars
    A^{i}:          The A coefficient matrix associate to integer vars
    x^{i}:          The integer vars
    sense^{i}:      The sense of the constraints to the integer vars, per constraint
    b_coef^{i}:     The coefficients of the right-hand side for the integer vars
    """
    # Variables are constraints of block
    # Constraints are unfixed variables of block and the parent model.
    #
    unfixed_c_vars = set()
    unfixed_b_vars = set()
    unfixed_i_vars = set()
    if not fixed and not unfixed:
        # If neither set was specified, then treat all variables as local
        unfixed = True
    elif unfixed:
        unfixed_vars = set()
        for v in unfixed:
            if v.is_continuous():
                unfixed_vars = 'unfixed_c_vars'
            if v.is_binary():
                unfixed_vars = 'unfixed_b_vars'
            if v.is_integer():
                unfixed_vars = 'unfixed_i_vars'

            if v.is_indexed():
                for vardata in v.values():
                    eval(unfixed_vars).add(id(vardata))
            else:
                eval(unfixed_vars).add(id(v))
        fixed_vars = set()
        unfixed = False
    elif fixed:
        unfixed_vars = set()
        fixed_vars = set()
        for v in fixed:
            if v.is_indexed():
                for vardata in v.values():
                    fixed_vars.add(id(vardata))
            else:
                fixed_vars.add(id(v))
        unfixed = False


    A_c = {}
    x_c = {}
    sense_c = {}
    b_coef_c = {}
    A_b = {}
    x_b = {}
    sense_b = {}
    b_coef_b = {}
    A_i = {}
    x_i = {}
    sense_i = {}
    b_coef_i = {}

    for data in block.component_objects(Constraint, active=True):
        A = {}
        sense = {}
        b_coef = {}
        name = data.getname(relative_to=block)
        for ndx in data:
            con = data[ndx]
            body_terms = generate_standard_repn(con.body, compute_values=True)
            for var, coef in zip(body_terms.linear_vars, body_terms.linear_coefs):
                if id(var) in unfixed_c_vars:
                    A = 'A_c'
                if id(var) in unfixed_b_vars:
                    A = 'A_b'
                if id(var) in unfixed_i_vars:
                    A = 'A_i'
                eval(A).setdefault(var.name, {}).setdefault(var.index(), []).append(
                                                Bunch(coef=coef, var=var, ndx=ndx))

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

            b_coef = {}
            if not con.equality:
                #
                # Inequality constraint
                # lower <= body <= upper
                # if both lower_terms and upper_terms are not None
                #
                if not(upper_terms is None):
                    #
                    # body <= upper
                    #
                    b_coef[name, 'g', ndx] = upper_terms.constant - body_terms.constant
                if not(lower_terms is None):
                    #
                    # lower <= body
                    #
                    b_coef[name, 'l', ndx] = lower_terms.constant - body_terms.constant
            else:
                #
                # Equality constraint
                #
                b_coef[name, 'e', ndx] = lower_terms.constant - body_terms.constant

                sense[name,ndx] = 'e'
            elif not(con.lower is None):
                sense[name,ndx] = 'l'
            elif not(con.upper is None):
                sense[name,ndx] = 'g'
