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
pao.pyomo.plugins

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
from ..components import SubModel
from numpy import zeros, array
from collections import OrderedDict


def collect_global_vars(model):
    """
    Collect vars on the model
    """
    c_var_ids = set()
    b_var_ids = set()
    i_var_ids = set()

    all_vars = dict()
    for v in model.component_objects(Var, descend_into=False, sort=True):
        if v.is_indexed():
            for vardata in v.values():
                all_vars[id(vardata)] = vardata
                if v.is_continuous():
                    c_var_ids.add(id(vardata))
                if v.is_binary():
                    b_var_ids.add(id(vardata))
                if v.is_integer():
                    i_var_ids.add(id(vardata))
        else:
            all_vars[id(v)] = v
            if v.is_continuous():
                c_var_ids.add(id(v))
            if v.is_binary():
                b_var_ids.add(id(v))
            if v.is_integer():
                i_var_ids.add(id(v))

    return all_vars, c_var_ids, b_var_ids, i_var_ids


def collect_fixed_vars(block):
    """
    Collect fixed vars on the block
    """
    fixed_var_ids = set()
    if hasattr(block, '_fixed'):
        for v in block._fixed:
            if v.is_indexed():
                for vardata in v.values():
                    fixed_var_ids.add(id(vardata))
            else:
                fixed_var_ids.add(id(v))

    return fixed_var_ids


def collect_bilevel_objective_vector_representation(block, all_vars, c_var_ids, b_var_ids, i_var_ids, fixed_var_ids,
                                                    standard_form=True):
    """
    Collects the terms for the objective
    for the given local variable scope, per variable type (c:continuous,
    b:binary, i:integer). Nested Blocks (or SubModels) are not considered
    within local scope.

    Arguments:
        block: A Pyomo Model, Block, or SubModel that will be processed to
                collect the objective terms.
        all_vars: An iterable, sorted object with Variable and VarData values.
        c_var_ids: Unique id for vars that are continuous and not fixed on this block.
        b_var_ids: Unique id for vars that are binary and not fixed on this block.
        i_var_ids: Unique id for vars that are integer and not fixed on this block.
        fixed_var_ids: Unique id for vars that are fixed on this block.
        standard_form: Whether to put the model into stanndard form representation (max for subproblem; min for model).

    Returns: Tuple with the following values:
    C^{c}:         The C coefficient scalar associate to continuous vars
    C_q^{c}:       The C coefficient vector associate to continuous vars in bilinear terms
    C^{b}:         The C coefficient scalar associate to binary vars
    C_q^{b}:       The C coefficient vector associate to binary vars in bilinear terms
    C^{i}:         The C coefficient scalar associate to integer vars
    C_q^{i}:       The C coefficient vector associate to integer vars in bilinear terms
    C^{f}:         The C coefficient scalar associate to fixed vars
    C_q^{f}:       The C coefficient vector associate to fixed vars in bilinear terms
    C_constant:    The C coefficient vector associate to constant terms
    """

    # number of variables
    _c = len(all_vars)

    # C coefficient for linear terms, continuous var
    C_c = {k: 0. for k in c_var_ids}

    # C coefficients for bilinear terms, for the continuous var
    C_c_q = {k: zeros(_c) for k in c_var_ids}

    # C coefficient for linear terms, binary var
    C_b = {k: 0. for k in b_var_ids}

    # C coefficients for bilinear terms, for the binary var
    C_b_q = {k: zeros(_c) for k in b_var_ids}

    # C coefficient for linear terms, integer var
    C_i = {k: 0. for k in i_var_ids}

    # C coefficients for bilinear terms, for the integer var
    C_i_q = {k: zeros(_c) for k in i_var_ids}

    # C coefficient for linear terms, fixed var in submodel
    C_f = {k: 0. for k in fixed_var_ids}

    # C coefficients for bilinear terms, fixed var in submodel
    C_f_q = {k: zeros(_c) for k in fixed_var_ids}

    C_constant = 0.

    nobj = 0
    for odata in block.component_objects(Objective, active=True):
        for ndx in odata:
            o_terms = generate_standard_repn(odata[ndx].expr, compute_values=False)
            C_constant = o_terms.constant
            _sign = 1.
            if odata[ndx].sense == minimize: # swith to a maximize lower-level
                if standard_form and not block.parent_block() is None:
                    _sign = -1.
            if odata[ndx].sense == maximize: # switch to a minimize upper-level
                if standard_form and id(block) == id(block.root_block()):
                    _sign = -1.
            for var, coef in zip(o_terms.linear_vars, o_terms.linear_coefs):
                _vid = id(var)
                if _vid in c_var_ids:
                    C_c[_vid] = _sign * coef
                if _vid in b_var_ids:
                    C_b[_vid] = _sign * coef
                if _vid in i_var_ids:
                    C_i[_vid] = _sign * coef
                if _vid in fixed_var_ids:
                    C_f[_vid] = _sign * coef

            for (var1, var2), coef in zip(o_terms.quadratic_vars, o_terms.quadratic_coefs):
                for (var, fixed) in [(var1, var2), (var2, var1)]:
                    _vid = id(var)
                    _col = list(all_vars.keys()).index(id(fixed)) if (id(fixed) in fixed_var_ids or id(var) in fixed_var_ids) else None
                    if not _col is None:
                        if _vid in c_var_ids:
                            C_c_q[_vid][_col] = _sign * coef
                        if _vid in b_var_ids:
                            C_b_q[_vid][_col] = _sign * coef
                        if _vid in i_var_ids:
                            C_i_q[_vid][_col] = _sign * coef
                        if _vid in fixed_var_ids:
                            C_f_q[_vid][_col] = _sign * coef
                    else:
                        raise RuntimeError(
                            "Error in matrix representation of bilevel Submodel block.  Bilinear expressions using"
                            "variables in the same bilevel upper- or lower-level.")
            nobj += 1
    if nobj == 0:
        raise RuntimeError("Error in vector representation of bilevel Submodel block.  No objective expression.")
    if nobj > 1:
        raise RuntimeError("Error in vector representation of bilevel Submodel block.  Multiple objective expressions.")

    return C_c, C_c_q, C_b, C_b_q, C_i, C_i_q, C_f, C_f_q, C_constant

def collect_bilevel_constraint_matrix_representation(block, all_vars, c_var_ids, b_var_ids, i_var_ids, fixed_var_ids,
                                                     standard_form=True):
    """
    Collects the terms for each ==, >= and <= constraints
    for the given local variable scope, per variable type (c:continuous,
    b:binary, i:integer). Nested Blocks (or SubModels) are not considered
    within local scope.

    TODO: The way the coo_matrix is constructed (by casting a numpy matrix)
    TODO: may be resource intensive. If that is the case, we will want to update the code
    TODO: as follows:
    # Constructing a coo_matrix using ijv format
    row  = np.array([r1, ...., rN])
    col  = np.array([c1, ...., cN])
    data = np.array([d1, ...., dN])
    _r = len(cons_sense_rhs)
    _c = len(all_vars)
    coo_matrix((data, (row, col)), shape=(_r, _c))

    Arguments:
        block: A Pyomo Model, Block, or SubModel that will be processed to
                collect terms for Ax <= b, Ax >= b, and Ax == b.
        all_vars: An iterable, sorted object with Variable and VarData values.
        c_var_ids: Unique id for vars that are continuous and not fixed on this block.
        b_var_ids: Unique id for vars that are binary and not fixed on this block.
        i_var_ids: Unique id for vars that are integer and not fixed on this block.
        fixed_var_ids: Unique id for vars that are fixed on this block.
        standard_form: Whether to put the model into stanndard form representation (<=, ==).

    Returns: Tuple with the following values:
    A^{c}:         The A coefficient vector associate to continuous vars
    A_q^{c}:       The A coefficient matrix associate to continuous vars in bilinear terms
    A^{b}:         The A coefficient vector associate to binary vars
    A_q^{b}:       The A coefficient matrix associate to binary vars in bilinear terms
    A^{i}:         The A coefficient vector associate to integer vars
    A_q^{i}:       The A coefficient matrix associate to integer vars in bilinear terms
    A^{f}:         The A coefficient vector associate to fixed vars
    A_q^{f}:       The A coefficient matrix associate to fixed vars in bilinear terms
    cons_sense_rhs: Unique id for constraints on the block.
    """

    # number of constraints
    _r = 0
    for data in block.component_map(Constraint, active=True).itervalues():
        _r += len(data)
        if data.has_lb() and data.has_ub():
            _r += len(data)

    # number of variables
    _c = len(all_vars)

    # A coefficients for linear terms, continuous var
    A_c = {k: zeros(_r) for k in c_var_ids}

    # A coefficients for bilinear terms, for the continuous var
    A_c_q = {k: zeros((_r, _c)) for k in c_var_ids}

    # A coefficients for linear terms, binary var
    A_b = {k: zeros(_r) for k in b_var_ids}

    # A coefficients for bilinear terms, for the binary var
    A_b_q = {k: zeros((_r, _c)) for k in b_var_ids}

    # A coefficients for linear terms, integer var
    A_i = {k: zeros(_r) for k in i_var_ids}

    # A coefficients for bilinear terms, for the integer var
    A_i_q = {k: zeros((_r, _c)) for k in i_var_ids}

    # A coefficients for linear terms, fixed var in submodel
    A_f = {k: zeros(_r) for k in fixed_var_ids}

    # A coefficients for bilinear terms, fixed var in submodel
    A_f_q = {k: zeros((_r, _c)) for k in fixed_var_ids}

    cons_sense_rhs = OrderedDict()
    for c in block.component_objects(Constraint, active=True, descend_into=True):
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
                cons_sense_rhs[id(con), 'e'] = lower_terms.constant - body_terms.constant
            else:
                if not (lower_terms is None):
                    if standard_form:
                        sense.append('g->l')
                        cons_sense_rhs[id(con), 'g->l'] = -(lower_terms.constant - body_terms.constant)
                    else:
                        sense.append('g')
                        cons_sense_rhs[id(con), 'g'] = lower_terms.constant - body_terms.constant
                if not (upper_terms is None):
                    sense.append('l')
                    cons_sense_rhs[id(con), 'l'] = upper_terms.constant - body_terms.constant
            for s in sense:
                _sign = 1.
                if s=='g->l':
                    _sign = -1.
                _row = list(cons_sense_rhs.keys()).index((_cid, s))

                for var, coef in zip(body_terms.linear_vars, body_terms.linear_coefs):
                    _vid = id(var)
                    if _vid in c_var_ids:
                        A_c[_vid][_row] = _sign*coef
                    if _vid in b_var_ids:
                        A_b[_vid][_row] = _sign*coef
                    if _vid in i_var_ids:
                        A_i[_vid][_row] = _sign*coef
                    if _vid in fixed_var_ids:
                        A_f[_vid][_row] = _sign*coef

                for (var1, var2), coef in zip(body_terms.quadratic_vars, body_terms.quadratic_coefs):
                    for (var, fixed) in [(var1, var2), (var2, var1)]:
                        _vid = id(var)
                        _col = list(all_vars.keys()).index(id(fixed)) if (id(fixed) in fixed_var_ids or id(var) in fixed_var_ids) else None
                        if not _col is None:
                            if _vid in c_var_ids:
                                A_c_q[_vid][_row, _col] = _sign*coef
                            if _vid in b_var_ids:
                                A_b_q[_vid][_row, _col] = _sign*coef
                            if _vid in i_var_ids:
                                A_i_q[_vid][_row, _col] = _sign*coef
                            if _vid in fixed_var_ids:
                                A_f_q[_vid][_row, _col] = _sign*coef
                        else:
                            raise RuntimeError(
                                "Error in matrix representation of bilevel Submodel block.  Bilinear expressions using"
                                " variables in the same bilevel upper- or lower-level.")

    A_c_q = {k: coo_matrix(v) for k, v in A_c_q.items()}
    A_b_q = {k: coo_matrix(v) for k, v in A_b_q.items()}
    A_i_q = {k: coo_matrix(v) for k, v in A_i_q.items()}
    A_f_q = {k: coo_matrix(v) for k, v in A_f_q.items()}

    return A_c, A_c_q, A_b, A_b_q, A_i, A_i_q, A_f, A_f_q, cons_sense_rhs


class BilevelMatrixRepn():
    """
    A class that creates an object which contains all the matrix expressions
    to the Pyomo model.
    """

    def __init__(self, model, **kwds):
        # Put in Standard Form
        self._standard_form = kwds.pop('standard_form', True)

        # Pyomo ConcreteModel
        self._model = model

        # List of model block name and all Submodels
        self._all_models = {block.name for block in model.component_objects(SubModel)}
        self._all_models.add(model.name)

        # All variables on model, including var id sets for continuous (c), binary (b) and integer (i) variable types
        all_vars, c_var_ids, b_var_ids, i_var_ids = collect_global_vars(model)
        self._all_vars = all_vars
        self._c_var_ids = c_var_ids
        self._b_var_ids = b_var_ids
        self._i_var_ids = i_var_ids

        # All fixed variables that exist on specified SubModel blocks (these do not exist on the Model block)
        self._fixed_var_ids = {block.name: collect_fixed_vars(block) for block in model.component_objects(SubModel)}

        # A coefficients for linear terms, continuous var
        self._A_c = {k: dict() for k in self._all_models}
        # A coefficients for bilinear terms, for the continuous var
        self._A_c_q = {k: dict() for k in self._all_models}
        # A coefficients for linear terms, binary var
        self._A_b = {k: dict() for k in self._all_models}
        # A coefficients for bilinear terms, for the binary var
        self._A_b_q = {k: dict() for k in self._all_models}
        # A coefficients for linear terms, integer var
        self._A_i = {k: dict() for k in self._all_models}
        # A coefficients for bilinear terms, for the integer var
        self._A_i_q = {k: dict() for k in self._all_models}
        # A coefficients for linear terms, fixed var in SubModel
        self._A_f = {k: dict() for k in self._all_models - {model.name}}
        # A coefficients for bilinear terms, fixed var in SubModel
        self._A_f_q = {k: dict() for k in self._all_models - {model.name}}
        # All constraint ids, corresponding sense and rhs; includes two entries for constraint declarations LHS <= body <= RHS
        # that are not equality constraints
        self._cons_sense_rhs = {k: dict() for k in self._all_models}

        # C coefficients for linear terms, continuous var
        self._C_c = {k: dict() for k in self._all_models}
        # C coefficients for bilinear terms, for the continuous var
        self._C_c_q = {k: dict() for k in self._all_models}
        # C coefficients for linear terms, binary var
        self._C_b = {k: dict() for k in self._all_models}
        # C coefficients for bilinear terms, for the binary var
        self._C_b_q = {k: dict() for k in self._all_models}
        # C coefficients for linear terms, integer var
        self._C_i = {k: dict() for k in self._all_models}
        # C coefficients for bilinear terms, for the integer var
        self._C_i_q = {k: dict() for k in self._all_models}
        # C coefficients for linear terms, fixed var in SubModel
        self._C_f = {k: dict() for k in self._all_models - {model.name}}
        # C coefficients for bilinear terms, fixed var in SubModel
        self._C_f_q = {k: dict() for k in self._all_models - {model.name}}
        # C coefficients for constant terms
        self._C_constant = {k: dict() for k in self._all_models}

        self._preprocess()

    def cost_vectors(self, block, var):
        if id(block) != id(block.root_block()) and id(var) in self._fixed_var_ids[block.name]:
            C = (self._C_f.get(block.name)).get(id(var))
            C_q = (self._C_f_q.get(block.name)).get(id(var))
        else:
            if var.is_continuous():
                C = (self._C_c.get(block.name)).get(id(var))
                C_q = (self._C_c_q.get(block.name)).get(id(var))
            if var.is_binary():
                C = (self._C_b.get(block.name)).get(id(var))
                C_q = (self._C_c_q.get(block.name)).get(id(var))
            if var.is_integer():
                C = (self._C_i.get(block.name)).get(id(var))
                C_q = (self._C_i_q.get(block.name)).get(id(var))

        C_constant = self._C_constant.get(block.name)

        return C, C_q, C_constant  # returns coo_matrix for selected rows

    def coef_matrices(self, block, var, sense=None):
        if id(block) != id(block.root_block()) and id(var) in self._fixed_var_ids[block.name]:
            A = (self._A_f.get(block.name)).get(id(var))
            A_q = (self._A_f_q.get(block.name)).get(id(var))
        else:
            if var.is_continuous():
                A = (self._A_c.get(block.name)).get(id(var))
                A_q = (self._A_c_q.get(block.name)).get(id(var))
            if var.is_binary():
                A = (self._A_b.get(block.name)).get(id(var))
                A_q = (self._A_c_q.get(block.name)).get(id(var))
            if var.is_integer():
                A = (self._A_i.get(block.name)).get(id(var))
                A_q = (self._A_i_q.get(block.name)).get(id(var))

        # return the full matrices if the sign on the constraint is not specified
        cons_sense_rhs = self._cons_sense_rhs[block.name]
        b = [rhs for (_cid, _sense), rhs in cons_sense_rhs.items()]
        sign = [_sense if _sense != 'g->l' else 'l' for (_cid, _sense), rhs in cons_sense_rhs.items()]
        if sense is None or not sense in ['e', 'l', 'g','g->l']:
            return A, A_q, sign, array(b)

        # grouping 'l' (<=) and those 'g->l' transformed to (<=) together
        if sense=='l':
            sense = ['l','g->l']
        else:
            sense=[sense]

        # return partial matrices if the sign on the constraint is specified
        indices = list()
        b = list()
        sign = list()
        for (_cid, _sense), rhs in cons_sense_rhs.items():
            if _sense in sense:
                idx = list(cons_sense_rhs.keys()).index((_cid, _sense))
                indices.append(idx)
                b.append(rhs)
                if _sense == 'g->l':
                    sign.append('l')
                else:
                    sign.append(_sense)

        if len(indices) == 0:
            return array(indices), coo_matrix((0, 0)), sign, array(b) # return empty coo_matrix for A, A_q

        cons_sense_rhs = self._cons_sense_rhs[block.name]
        if len(indices) == len(cons_sense_rhs):
            return A, A_q, sign, array(b)  # return the full coo_matrix

        return A[indices], A_q.todok()[indices, :].tocoo(), sign, array(b)  # returns coo_matrix for selected rows

    def _preprocess(self):
        # Preprocess the main Model (upper-level)
        A_c, A_c_q, A_b, A_b_q, A_i, A_i_q, A_f, A_f_q, cons_sense_rhs = \
            collect_bilevel_constraint_matrix_representation(self._model, \
                                                             self._all_vars, \
                                                             self._c_var_ids, \
                                                             self._b_var_ids, \
                                                             self._i_var_ids, \
                                                             [], \
                                                             standard_form = self._standard_form)
        self._A_c[self._model.name] = A_c
        self._A_c_q[self._model.name] = A_c_q
        self._A_b[self._model.name] = A_b
        self._A_b_q[self._model.name] = A_b_q
        self._A_i[self._model.name] = A_i
        self._A_i_q[self._model.name] = A_i_q
        self._cons_sense_rhs[self._model.name] = cons_sense_rhs

        C_c, C_c_q, C_b, C_b_q, C_i, C_i_q, C_f, C_f_q, C_constant = \
            collect_bilevel_objective_vector_representation(self._model, \
                                                            self._all_vars, \
                                                            self._c_var_ids, \
                                                            self._b_var_ids, \
                                                            self._i_var_ids, \
                                                            [], \
                                                            standard_form = self._standard_form)
        self._C_c[self._model.name] = C_c
        self._C_c_q[self._model.name] = C_c_q
        self._C_b[self._model.name] = C_b
        self._C_b_q[self._model.name] = C_b_q
        self._C_i[self._model.name] = C_i
        self._C_i_q[self._model.name] = C_i_q
        self._C_constant[self._model.name] = C_constant

        # Preprocess each SubModel (lower-level(s))
        for block in self._model.component_objects(SubModel):
            A_c, A_c_q, A_b, A_b_q, A_i, A_i_q, A_f, A_f_q, cons_sense_rhs = \
                collect_bilevel_constraint_matrix_representation(block, \
                                                                 self._all_vars, \
                                                                 self._c_var_ids - self._fixed_var_ids[block.name], \
                                                                 self._b_var_ids - self._fixed_var_ids[block.name], \
                                                                 self._i_var_ids - self._fixed_var_ids[block.name], \
                                                                 self._fixed_var_ids[block.name], \
                                                                 standard_form = self._standard_form)
            self._A_c[block.name] = A_c
            self._A_c_q[block.name] = A_c_q
            self._A_b[block.name] = A_b
            self._A_b_q[block.name] = A_b_q
            self._A_i[block.name] = A_i
            self._A_i_q[block.name] = A_i_q
            self._A_f[block.name] = A_f
            self._A_f_q[block.name] = A_f_q
            self._cons_sense_rhs[block.name] = cons_sense_rhs

            C_c, C_c_q, C_b, C_b_q, C_i, C_i_q, C_f, C_f_q, C_constant = \
                collect_bilevel_objective_vector_representation(block, \
                                                                self._all_vars, \
                                                                self._c_var_ids - self._fixed_var_ids[block.name], \
                                                                self._b_var_ids - self._fixed_var_ids[block.name], \
                                                                self._i_var_ids - self._fixed_var_ids[block.name], \
                                                                self._fixed_var_ids[block.name], \
                                                                standard_form = self._standard_form)
            self._C_c[block.name] = C_c
            self._C_c_q[block.name] = C_c_q
            self._C_b[block.name] = C_b
            self._C_b_q[block.name] = C_b_q
            self._C_i[block.name] = C_i
            self._C_i_q[block.name] = C_i_q
            self._C_f[block.name] = C_f
            self._C_f_q[block.name] = C_f_q
            self._C_constant[block.name] = C_constant
