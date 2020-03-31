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
from scipy.sparse import coo_matrix
from numpy import zeros

def collect_block_constraints(block):

    cons_sense_rhs = dict()
    for c in block.component_objects(Constraint, active=True, sort=True):
        for ndx in c:
            con = c[ndx]
            lower_terms = generate_standard_repn(con.lower, compute_values=False) \
                                                    if not con.lower is None else None
            upper_terms = generate_standard_repn(con.upper, compute_values=False) \
                                                    if not con.upper is None else None
            body_terms = generate_standard_repn(con.body, compute_values=True)
            if con.equality:
                cons_sense_rhs[id(con),'e'] = lower_terms.constant - body_terms.constant
            else:
                if not (lower_terms is None):
                    cons_sense_rhs[id(con),'g'] = lower_terms.constant - body_terms.constant
                if not (upper_terms is None):
                    cons_sense_rhs[id(con),'l'] = upper_terms.constant - body_terms.constant

    return cons_sense_rhs

def collect_block_vars(block):
    """
    Collect vars on the block
    """
    c_var_ids = set()
    b_var_ids = set()
    i_var_ids = set()
    fixed_var_ids = set()
    all_vars = dict()
    if hasattr(block,'_fixed'):
        for v in block._fixed:
            if v.is_indexed():
                for vardata in v.values():
                    fixed_var_ids.add(id(vardata))
                    all_vars[id(vardata)] = vardata
            else:
                fixed_var_ids.add(id(v))

    root = block.root_block()
    for v in root.component_objects(Var, descend_into=False, sort=True):
        if id(v) not in fixed_var_ids:
            if v.is_indexed():
                for vardata in v.values():
                    if v.is_continuous():
                        c_var_ids.add(id(vardata))
                    if v.is_binary():
                        b_var_ids.add(id(vardata))
                    if v.is_integer():
                        i_var_ids.add(id(vardata))
                    all_vars[id(vardata)] = vardata
            else:
                if v.is_continuous():
                    c_var_ids.add(id(v))
                if v.is_binary():
                    b_var_ids.add(id(v))
                if v.is_integer():
                    i_var_ids.add(id(v))
                all_vars[id(v)] = v
        else:
            if v.is_indexed():
                for vardata in v.values():
                    all_vars[id(vardata)] = vardata
            else:
                all_vars[id(v)] = v


    return c_var_ids, b_var_ids, i_var_ids, fixed_var_ids, all_vars


def collect_bilevel_matrix_representation(block):#, c_var_ids, b_var_ids, i_var_ids, fixed_var_ids):
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

    c_var_ids, b_var_ids, i_var_ids, fixed_var_ids, all_vars = \
        collect_block_vars(block)

    cons_sense_rhs = collect_block_constraints(block)

    _r = len(cons_sense_rhs) # sum(len(v) for v in cons_sense_rhs.itervalues())
    _c = len(all_vars)

    # A coefficients for linear terms, continuous var
    A_c = {k: zeros((_r,_c)) for k in c_var_ids}

    # A coefficients for bilinear terms, for the continuous var
    A_c_q = {k: zeros((_r,_c)) for k in c_var_ids}

    # A coefficients for linear terms, binary var
    A_b = {k: zeros((_r,_c)) for k in b_var_ids}

    # A coefficients for bilinear terms, for the binary var
    A_b_q = {k: zeros((_r,_c)) for k in b_var_ids}

    # A coefficients for linear terms, integer var
    A_i = {k: zeros((_r,_c)) for k in i_var_ids}

    # A coefficients for bilinear terms, for the integer var
    A_i_q = {k: zeros((_r,_c)) for k in i_var_ids}

    # A coefficients for linear terms, fixed var in submodel
    A_f = {k: zeros((_r,_c)) for k in fixed_var_ids}

    # A coefficients for bilinear terms, fixed var in submodel
    A_f_q = {k: zeros((_r,_c)) for k in fixed_var_ids}

    # sense_c = {}
    # b_coef_c = {}
    # sense_b = {}
    # b_coef_b = {}
    # sense_i = {}
    # b_coef_i = {}
    # sense_f = {}
    # b_coef_f = {}

    for c in block.component_objects(Constraint, active=True, sort=True):
        sense = list()
        for ndx in c:
            con = c[ndx]
            _cid = id(con)

            lower_terms = generate_standard_repn(con.lower, compute_values=False) \
                                                    if not con.lower is None else None
            upper_terms = generate_standard_repn(con.upper, compute_values=False) \
                                                    if not con.upper is None else None
            body_terms = generate_standard_repn(con.body, compute_values=True)
            if con.equality:
                sense.append('e')
            else:
                if not (lower_terms is None):
                    sense.append('g')
                if not (upper_terms is None):
                    sense.append('l')
            for s in sense:
                _row = list(cons_sense_rhs.keys()).index((_cid,s))

                for var, coef in zip(body_terms.linear_vars, body_terms.linear_coefs):
                    _vid = id(var)
                    _col = list(all_vars.keys()).index(_vid)
                    if _vid in c_var_ids:
                        A_c[_vid][_row, _col] = coef
                    if _vid in b_var_ids:
                        A_b[_vid][_row, _col] = coef
                    if _vid in i_var_ids:
                        A_i[_vid][_row, _col] = coef
                    if _vid in fixed_var_ids:
                        A_f[_vid][_row, _col] = coef

                for (var1,var2), coef in zip(body_terms.quadratic_vars, body_terms.quadratic_coefs):
                    for (var,fixed) in [(var1, var2),(var2, var1)]:
                        _vid = id(var)
                        _col = list(all_vars.keys()).index(id(fixed))
                        if _vid in c_var_ids:
                            A_c_q[_vid][_row, _col] = coef
                        if _vid in b_var_ids:
                            A_b_q[_vid][_row, _col] = coef
                        if _vid in i_var_ids:
                            A_i_q[_vid][_row, _col] = coef
                        if _vid in fixed_var_ids:
                            A_f_q[_vid][_row, _col] = coef

    return





# class BilevelMatrixRepn():
#     """
#     A class that creates an object which contains all the matrix expressions
#     to the Pyomo model.
#     """
#
#     def __init__(self, **kwds):
#         iter = 0
#
#         A_c = dict()
#         A_c_q
#

# constraint id
