import pyomo.environ as pe
from pyomo.core.base.block import _BlockData, ScalarBlock
from pyomo.core.base.var import _GeneralVarData, ScalarVar, IndexedVar
from typing import Sequence, Optional, Iterable, List
import math
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.contrib.appsi.base import PersistentBase
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _GeneralConstraintData, IndexedConstraint
from .utils import ComponentHasher
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData


class Dual(PersistentBase):
    def __init__(self, fixed_vars: Optional[Iterable[_GeneralVarData]] = None):
        super().__init__(only_child_vars=False)
        self._dual = ScalarBlock(concrete=True)
        self._setup_dual()
        self._grad_lag_map = pe.ComponentMap()
        self._lagrangian_terms = dict()
        if fixed_vars is None:
            self._fixed_vars = ComponentSet()
        else:
            self._fixed_vars = ComponentSet(fixed_vars)
        self._old_obj = None
        self._old_obj_vars = list()

    def _setup_dual(self):
        self._dual.grad_lag_set = pe.Set(dimen=1)
        self._dual.eq_dual_set = pe.Set(dimen=1)
        self._dual.ineq_dual_set = pe.Set(dimen=1)

        self._dual.eq_duals = IndexedVar(self._dual.eq_dual_set)
        self._dual.ineq_duals = IndexedVar(self._dual.ineq_dual_set, bounds=(0, None))

        self._dual.grad_lag = IndexedConstraint(self._dual.grad_lag_set)

        self._dual.objective = pe.Objective(expr=0, sense=pe.maximize)

    def _set_dual_obj(self):
        self._dual.objective.expr = sum(self._lagrangian_terms.values())
        raise NotImplementedError('Still need to make sure self._lagrangian_terms gets updated correctly')

    def dual(self, model: _BlockData):
        if model is not self._model:
            self._dual = ScalarBlock(concrete=True)
            self._setup_dual()
            self.set_instance(model)
        else:
            self.update()
        self._set_dual_obj()
        return self._dual

    def _update_var_bounds(self, v):
        v_lb, v_ub = v.bounds
        lb_hasher = ComponentHasher(v, 'lb')
        ub_hasher = ComponentHasher(v, 'ub')

        for v_bound, hasher in [(v_lb, lb_hasher), (v_ub, ub_hasher)]:
            if v_bound is None:
                if hasher in self._grad_lag_map[v]:
                    self._grad_lag_map[v].pop(hasher)
                    del self._dual.ineq_duals[hasher]
                    self._dual.ineq_dual_set.remove(hasher)
                    self._lagrangian_terms.pop(hasher)
            else:
                if hasher not in self._grad_lag_map[v]:
                    self._dual.ineq_dual_set.add(hasher)
                if hasher.bound == 'lb':
                    e = self._dual.ineq_duals[hasher] * (v_lb - v)
                else:
                    e = self._dual.ineq_duals[hasher] * (v - v_ub)
                if hasher in self._grad_lag_map[v]:
                    self._grad_lag_map[v][hasher] = reverse_sd(e)[v]
                    self._lagrangian_terms[hasher] = e
                else:
                    self._process_lagrangian_term(e, [v], hasher)

    def _add_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            if v in self._fixed_vars:
                continue
            if v.fixed:
                continue
            self._grad_lag_map[v] = dict()
            v_hasher = ComponentHasher(v, None)
            self._dual.grad_lag_set.add(v_hasher)
            self._dual.grad_lag[v_hasher] = (0, 0)
            self._update_var_bounds(v)

    def _remove_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            if v in self._fixed_vars:
                continue

            v_hasher = ComponentHasher(v, None)
            lb_hasher = ComponentHasher(v, 'lb')
            ub_hasher = ComponentHasher(v, 'ub')

            if lb_hasher in self._dual.ineq_dual_set:
                del self._dual.ineq_duals[lb_hasher]
                self._dual.ineq_dual_set.remove(lb_hasher)
                self._lagrangian_terms.pop(lb_hasher)

            if ub_hasher in self._dual.ineq_dual_set:
                del self._dual.ineq_duals[ub_hasher]
                self._dual.ineq_dual_set.remove(ub_hasher)
                self._lagrangian_terms.pop(ub_hasher)

            if v_hasher in self._dual.grad_lag_set:
                # the variable may have been removed from these already if it was fixed
                # sometime since calling add_variables
                del self._dual.grad_lag[v_hasher]
                self._dual.grad_lag_set.remove(v_hasher)
                self._grad_lag_map.pop(v)

    def _regenerate_grad_lag_for_var(self, v):
        v_hasher = ComponentHasher(v, None)
        new_body = 0
        for c, der in self._grad_lag_map[v].items():
            new_body += der
        self._dual.grad_lag[v_hasher] = (new_body, 0)

    def _add_params(self, params: List[_ParamData]):
        pass

    def _remove_params(self, params: List[_ParamData]):
        pass

    def _process_lagrangian_term(self, expr, variables, c_hasher):
        self._lagrangian_terms[c_hasher] = expr
        ders = reverse_sd(expr)
        for v in variables:
            if v in self._fixed_vars:
                continue
            if v.fixed:
                continue
            v_hasher = ComponentHasher(v, None)
            orig_body = self._dual.grad_lag[v_hasher].body
            new_body = orig_body + ders[v]
            self._dual.grad_lag[v_hasher] = (new_body, 0)
            self._grad_lag_map[v][c_hasher] = ders[v]

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        for c in cons:
            if c.equality or (c.lb is not None and c.ub is not None and c.lb == c.ub):
                c_hasher = ComponentHasher(c, None)
                self._dual.eq_dual_set.add(c_hasher)
                e = self._dual.eq_duals[c_hasher] * (c.body - c.lb)
                self._process_lagrangian_term(e, self._vars_referenced_by_con[c], c_hasher)
            else:
                if c.lb is not None:
                    c_hasher = ComponentHasher(c, 'lb')
                    self._dual.ineq_dual_set.add(c_hasher)
                    e = self._dual.ineq_duals[c_hasher] * (c.lb - c.body)
                    self._process_lagrangian_term(e, self._vars_referenced_by_con[c], c_hasher)
                if c.ub is not None:
                    c_hasher = ComponentHasher(c, 'ub')
                    self._dual.ineq_dual_set.add(c_hasher)
                    e = self._dual.ineq_duals[c_hasher] * (c.body - c.ub)
                    self._process_lagrangian_term(e, self._vars_referenced_by_con[c], c_hasher)

    def _removal_helper(self, c_hasher, variables):
        if c_hasher in self._dual.eq_dual_set or c_hasher in self._dual.ineq_dual_set:
            self._lagrangian_terms.pop(c_hasher)
            for v in variables:
                if v in self._grad_lag_map:
                    self._grad_lag_map[v].pop(c_hasher)
            if c_hasher in self._dual.eq_dual_set:
                del self._dual.eq_duals[c_hasher]
                self._dual.eq_dual_set.remove(c_hasher)
            else:
                del self._dual.ineq_duals[c_hasher]
                self._dual.ineq_dual_set.remove(c_hasher)

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        affected_variables = ComponentSet()
        for c in cons:
            for c_hasher in [
                ComponentHasher(c, None),
                ComponentHasher(c, 'lb'),
                ComponentHasher(c, 'ub')
            ]:
                variables = self._vars_referenced_by_con[c]
                self._removal_helper(c_hasher, variables)
                affected_variables.update(variables)

        for v in affected_variables:
            if v in self._grad_lag_map:
                self._regenerate_grad_lag_for_var(v)

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        raise NotImplementedError('Dual does not support SOS constraints')

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        raise NotImplementedError('Dual does not support SOS constraints')

    def _set_objective(self, obj: _GeneralObjectiveData):
        if obj.sense != pe.minimize:
            raise NotImplementedError('Dual does not support maximization problems yet')
        if self._old_obj is not None:
            old_hasher = ComponentHasher(self._old_obj, None)
            self._lagrangian_terms.pop(old_hasher)
            for v in self._old_obj_vars:
                if v in self._grad_lag_map:
                    self._grad_lag_map[v].pop(ComponentHasher(self._old_obj, None))
                    self._regenerate_grad_lag_for_var(v)

        hasher = ComponentHasher(obj, None)
        e = obj.expr
        self._lagrangian_terms[hasher] = e
        self._process_lagrangian_term(e, self._vars_referenced_by_obj, hasher)
        self._old_obj = obj
        self._old_obj_vars = self._vars_referenced_by_obj

    def _update_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            if v in self._fixed_vars:
                continue
            if v.fixed:
                self._remove_variables([v])
            else:
                if v in self._grad_lag_map:
                    self._update_var_bounds(v)
                    self._regenerate_grad_lag_for_var(v)
                else:
                    # the variable was fixed but is now unfixed
                    # the persistent base class will remove and add any
                    # constraints that involve this variable
                    self._add_variables([v])

    def update_params(self):
        pass
