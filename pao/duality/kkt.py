import pyomo.environ as pe
from pyomo.core.base.block import _BlockData, ScalarBlock
from pyomo.core.base.var import _GeneralVarData, ScalarVar, IndexedVar
from typing import Sequence, Optional, Iterable
import math
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd


class ComponentHasher(object):
    def __init__(self, var, bound):
        self.comp = var
        self.bound = bound
        
    def __eq__(self, other):
        if not isinstance(other, ComponentHasher):
            return False
        if self.comp is other.comp and self.bound is other.bound:
            return True
        else:
            return False
        
    def __hash__(self):
        return hash((id(self.comp), self.bound))

    def __repr__(self):
        first = str(self.comp)
        if self.bound is None:
            return first
        else:
            second = str(self.bound)
            res = str((first, second))
            res = res.replace("'", "")
            return res

    def __str__(self):
        return self.__repr__()


def convert_integers_to_binaries(
    model: _BlockData,
    integer_vars: Iterable[_GeneralVarData],
) -> _BlockData:
    parent = ScalarBlock(concrete=True)
    parent.int_set = pe.Set(dimen=1)
    parent.bin_set = pe.Set(dimen=2)

    parent.bins = IndexedVar(parent.bin_set, domain=pe.Binary, bounds=(0, 1))
    parent.int_cons = pe.Constraint(parent.int_set)

    for iv in integer_vars:
        if not iv.is_integer():
            raise ValueError(f'{iv} is not an integer variable!')
        iv_hasher = ComponentHasher(iv, None)
        lb, ub = iv.bounds
        iv.domain = pe.Reals
        iv.setlb(lb)
        iv.setub(ub)
        parent.int_set.add(iv_hasher)
        n = math.ceil(math.log2(max(abs(lb), abs(ub)) + 1) + 1)
        expr = 0
        for ndx in range(n-1):
            parent.bin_set.add((iv_hasher, ndx))
            bv = parent.bins[iv_hasher, ndx]
            expr += 2**ndx * bv
        if lb < 0:
            parent.bin_set.add((iv_hasher, n-1))
            bv = parent.bins[iv_hasher, n-1]
            expr -= 2**(n-1) * bv
        parent.int_cons[iv_hasher] = iv == expr

    parent.block = model

    return parent


def convert_binary_domain_to_constraint(
    model: _BlockData,
    binary_vars: Iterable[_GeneralVarData],
) -> _BlockData:
    parent = ScalarBlock(concrete=True)
    parent.bin_set = pe.Set(dimen=1)
    parent.bin_cons = pe.Constraint(parent.bin_set)

    for bv in binary_vars:
        if not bv.is_binary():
            raise ValueError(f'{bv} is not binary variable!')
        bv_hasher = ComponentHasher(bv, None)
        bv.domain = pe.Reals
        bv.setlb(0)
        bv.setub(1)
        parent.bin_set.add(bv_hasher)
        parent.bin_cons[bv_hasher] = bv - bv**2 <= 0

    parent.block = model

    return parent


def _get_vars_in_expr(expr, all_vars, bin_vars, int_vars):
    for v in identify_variables(expr, include_fixed=False):
        all_vars.add(v)
        if v.is_binary():
            bin_vars.add(v)
        elif v.is_integer():
            int_vars.add(v)


def construct_kkt(
    model: _BlockData,
    variables: Optional[Sequence[_GeneralVarData]] = None
) -> _BlockData:
    all_vars = ComponentSet()
    int_vars = ComponentSet()
    bin_vars = ComponentSet()

    all_cons = ComponentSet()
    obj = None

    for _obj in model.component_data_objects(
            pe.Objective, active=True, descend_into=True
    ):
        if obj is not None:
            raise ValueError('found multiple active objectives')
        obj = _obj
        if variables is None:
            _get_vars_in_expr(obj.expr, all_vars, bin_vars, int_vars)

    for con in model.component_data_objects(
            pe.Constraint, active=True, descend_into=True
    ):
        if con.lb is not None or con.ub is not None:
            all_cons.add(con)
            if variables is None:
                _get_vars_in_expr(con.body, all_vars, bin_vars, int_vars)

    if variables is not None:
        for v in variables:
            all_vars.add(v)
            if v.is_binary():
                bin_vars.add(v)
            elif v.is_integer():
                int_vars.add(v)

    if len(int_vars) > 0:
        model = convert_integers_to_binaries(model, int_vars)
        for v in model.bins.values():
            bin_vars.add(v)

    if len(bin_vars) > 0:
        model = convert_binary_domain_to_constraint(model, bin_vars)

    parent = ScalarBlock(concrete=True)
    parent.block = model
    parent.eq_dual_set = pe.Set(dimen=1)
    parent.ineq_dual_set = pe.Set(dimen=1)
    parent.eq_duals = pe.Var(parent.eq_dual_set)
    parent.ineq_duals = pe.Var(parent.ineq_dual_set, bounds=(0, None))

    if obj is None:
        raise ValueError('no active objective was found')
    if obj.sense == pe.minimize:
        lagrangian = obj.expr
    else:
        lagrangian = -obj.expr

    for c in all_cons:
        if c.equality or (c.lb is not None and c.ub is not None and c.lb == c.ub):
            c_hasher = ComponentHasher(c, None)
            parent.eq_dual_set.add(c_hasher)
            lagrangian += parent.eq_duals[c_hasher] * (c.body - c.lb)
        else:
            if c.lb is not None:
                c_hasher = ComponentHasher(c, 'lb')
                parent.ineq_dual_set.add(c_hasher)
                lagrangian += parent.ineq_duals[c_hasher] * (c.lb - c.body)
            if c.ub is not None:
                c_hasher = ComponentHasher(c, 'ub')
                parent.ineq_dual_set.add(c_hasher)
                lagrangian += parent.ineq_duals[c_hasher] * (c.body - c.ub)

    for v in all_vars:
        v_lb, v_ub = v.bounds
        if v_lb is not None:
            v_hasher = ComponentHasher(v, 'lb')
            parent.ineq_dual_set.add(v_hasher)
            lagrangian += parent.ineq_duals[v_hasher] * (v_lb - v)
        if v_ub is not None:
            v_hasher = ComponentHasher(v, 'ub')
            parent.ineq_dual_set.add(v_hasher)
            lagrangian += parent.ineq_duals[v_hasher] * (v - v_ub)

    parent.grad_lag_set = pe.Set(dimen=1)
    parent.grad_lag = pe.Constraint(parent.grad_lag_set)
    ders = reverse_sd(lagrangian)
    for v in all_vars:
        v_hasher = ComponentHasher(v, None)
        parent.grad_lag_set.add(v_hasher)
        parent.grad_lag[v_hasher] = ders[v] == 0

    parent.complimentarity = pe.Constraint(parent.ineq_dual_set)
    for c_hasher in parent.ineq_dual_set:
        c = c_hasher.comp
        bound = c_hasher.bound
        if bound == 'lb':
            if isinstance(c, _GeneralVarData):
                expr = c.lb - c
            else:
                expr = c.lb - c.body
        else:
            assert bound == 'ub'
            if isinstance(c, _GeneralVarData):
                expr = c - c.ub
            else:
                expr = c.body - c.ub
        parent.complimentarity[c_hasher] = parent.ineq_duals[c_hasher] * expr == 0

    return parent
