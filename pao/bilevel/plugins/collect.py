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


def collect_bilevel_matrix_representation(block):
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
    A^{f}:          The A coefficient matrix associate to fixed vars
    x^{f}:          The fixed vars
    sense^{f}:      The sense of the constraints to the fixed vars, per constraint
    b_coef^{f}:     The coefficients of the right-hand side for the fixed vars
    """
    #
    # Group like variables
    #
    c_vars = set()
    b_vars = set()
    i_vars = set()
    fixed_vars = set()

    if hasattr(block,'_fixed'):
        for v in block._fixed:
            if v.is_indexed():
                for vardata in v.values():
                    fixed_vars.add(id(vardata))
            else:
                fixed_vars.add(id(v))
    for v in block.component_objects(Var, descend_into=False):
        if v.is_indexed():
            for vardata in v.values():
                if v.is_continuous():
                    c_vars.add(id(vardata))
                if v.is_binary():
                    b_vars.add(id(vardata))
                if v.is_integer():
                    i_vars.add(id(vardata))
        else:
            if v.is_continuous():
                c_vars.add(id(v))
            if v.is_binary():
                b_vars.add(id(v))
            if v.is_integer():
                i_vars.add(id(v))


    A_c = {}
    sense_c = {}
    b_coef_c = {}
    A_b = {}
    sense_b = {}
    b_coef_b = {}
    A_i = {}
    sense_i = {}
    b_coef_i = {}
    A_f = {}
    sense_f = {}
    b_coef_f = {}

    def _set_sense_b(b_coef, sense, con, lower_terms, upper_terms, body_terms, var, ndx):

        if not con.equality:
            #
            # Inequality constraint
            # lower <= body <= upper
            # if both lower_terms and upper_terms are not None
            #
            if not (upper_terms is None):
                #
                # body <= upper
                #
                rhs = upper_terms.constant - body_terms.constant
                con_sense = 'l'
                b_coef.setdefault(var.name, {}).setdefault(var.index(), []).append(
                    Bunch(rhs=rhs, var=var, ndx=ndx))
            if not (lower_terms is None):
                #
                # lower <= body
                #
                rhs = lower_terms.constant - body_terms.constant
                con_sense = 'g'
                b_coef.setdefault(var.name, {}).setdefault(var.index(), []).append(
                    Bunch(rhs=rhs, var=var, ndx=ndx))
        else:
            #
            # Equality constraint
            #
            rhs = lower_terms.constant - body_terms.constant
            con_sense = 'e'
        b_coef.setdefault(var.name, {}).setdefault(var.index(), []).append(
                Bunch(rhs=rhs, var=var, ndx=ndx))
        sense.setdefault(var.name, {}).setdefault(var.index(), []).append(
                Bunch(sense=con_sense, var=var, ndx=ndx))

    for data in block.component_objects(Constraint, active=True):
        A = {}
        sense = {}
        b_coef = {}
        name = data.getname(relative_to=block)
        for ndx in data:
            con = data[ndx]
            lower_terms = generate_standard_repn(con.lower, compute_values=False) \
                                                    if not con.lower is None else None
            upper_terms = generate_standard_repn(con.upper, compute_values=False) \
                                                    if not con.upper is None else None
            body_terms = generate_standard_repn(con.body, compute_values=True)
            for var, coef in zip(body_terms.linear_vars, body_terms.linear_coefs):
                if id(var) in c_vars:
                    A_c.setdefault(var.name, {}).setdefault(var.index(), []).append(
                        Bunch(coef=coef, var=var, ndx=ndx))
                    _set_sense_b(b_coef_c, sense_c, con, lower_terms, upper_terms, body_terms, var, ndx)
                if id(var) in b_vars:
                    A_b.setdefault(var.name, {}).setdefault(var.index(), []).append(
                        Bunch(coef=coef, var=var, ndx=ndx))
                    _set_sense_b(b_coef_b, sense_b, con, lower_terms, upper_terms, body_terms, var, ndx)
                if id(var) in i_vars:
                    A_i.setdefault(var.name, {}).setdefault(var.index(), []).append(
                        Bunch(coef=coef, var=var, ndx=ndx))
                    _set_sense_b(b_coef_i, sense_i, con, lower_terms, upper_terms, body_terms, var, ndx)
                if id(var) in fixed_vars:
                    A_f.setdefault(var.name, {}).setdefault(var.index(), []).append(
                        Bunch(coef=coef, var=var, ndx=ndx))
                    _set_sense_b(b_coef_f, sense_f, con, lower_terms, upper_terms, body_terms, var, ndx)

    return (A_c, c_vars, sense_c, b_coef_c, A_b, b_vars, sense_b, b_coef_b, A_i, i_vars, sense_i, b_coef_i, A_f, fixed_vars, sense_f, b_coef_f)




