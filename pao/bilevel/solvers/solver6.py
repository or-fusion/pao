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
pao.bilevel.plugins.solver6

Declare the pao.bilevel.ccg solver.

"A Projection-Based Reformulation and Decomposition
Algorithm for Global Optimization of a Class of Mixed Integer
Bilevel Linear Programs"
D. Yue and J, Gao and B. Zeng and F. You
Journal of Global Optimization (2019) 73:27-57

"""

import time
import itertools
import pyutilib.misc
import pyomo.opt
from math import inf
import pyomo.common
from pyomo.mpec import complements
from pyomo.gdp import Disjunct
from pao.bilevel.solvers.solver_helpers import _check_termination_condition
from pao.bilevel.plugins.collect import BilevelMatrixRepn
from pao.bilevel.components import SubModel, varref, dataref
from pyomo.core import TransformationFactory, minimize, maximize, Block, Constraint, Objective, Var, Reals, ComplementarityList
from pyomo.core.expr.numvalue import value
from numpy import array, dot, sum

@pyomo.opt.SolverFactory.register('pao.bilevel.ccg',
                                  doc='Solver for projection-based reformulation and decomposition.')
class BilevelSolver5(pyomo.opt.OptSolver):
    """
    A solver for bilevel mixed-integer with both continuous and integer in upper
    and lower levels
    """

    def __init__(self, **kwds):
        kwds['type'] = 'pao.bilevel.ccg'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True
        self._k_max_iter = 10

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        self.results = []
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)
        self._upper_level_sense = minimize
        self._lower_level_sense = maximize

        # put problem into standard form
        for odata in self._instance.component_objects(Objective):
            if odata.parent_block() == self._instance:
                if odata.sense == maximize:
                    self._upper_level_sense = maximize
                    odata.set_value(-odata.expr)
                    odata.set_sense(minimize)
            if type(odata.parent_block()) == SubModel:
                if odata.sense == minimize:
                    self._lower_level_sense = minimize
                    odata.set_value(-odata.expr)
                    odata.set_sense(maximize)

    def _apply_solver(self):
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            solver = 'gurobi'

        # Step 1. Initialization
        LB = -inf
        UB = inf
        theta = 0.
        xi = 10e-3
        k = 0
        epsilon = 10e-3

        # matrix representation for bilevel problem
        matrix_repn = BilevelMatrixRepn(self._instance)

        # each lower-level problem
        submodel = [block for block in self._instance.component_objects(SubModel)][0]
        if len(submodel) != 1:
            raise Exception('Problem encountered, this is not a valid bilevel model for the solver.')
        self._instance.reclassify_component_type(submodel, Block)
        varref(submodel)
        dataref(submodel)

        all_vars = [var for (key, var) in matrix_repn._all_vars.items()]



        # get the variables that are fixed for the submodel (lower-level block)
        fixed_var_ids = matrix_repn._fixed_var_ids[submodel.name]
        fixed_vars = [var for (key, var) in matrix_repn._all_vars.items() if key in fixed_var_ids]

        c_var_ids = matrix_repn._c_var_ids - fixed_var_ids  # continuous variables in SubModel
        c_vars = [var for (key, var) in matrix_repn._all_vars.items() if key in c_var_ids]

        b_var_ids = matrix_repn._b_var_ids - fixed_var_ids  # binary variables in SubModel
        b_vars = [var for (key, var) in matrix_repn._all_vars.items() if key in b_var_ids]

        i_var_ids = matrix_repn._i_var_ids - fixed_var_ids  # integer variables in SubModel
        i_vars = [var for (key, var) in matrix_repn._all_vars.items() if key in i_var_ids]

        sub_cons_ids = set(matrix_repn._cons_sense_rhs[submodel.name].keys())
        sub_cons_sense_rhs = matrix_repn._cons_sense_rhs[submodel.name]

        _c_var_bounds_rule = lambda m, k: c_var_ids[k].bounds
        _iter_c = {k: Var(c_var_ids, bounds=_c_var_bounds_rule, domain=Reals)}
        _iter_c_tilde = {k: Var(c_var_ids, bounds=_c_var_bounds_rule, domain=Reals)}
        _iter_b = {k: Var(b_var_ids, bounds=_c_var_bounds_rule)}
        _iter_i = {k: Var(i_var_ids, bounds=_c_var_bounds_rule)}

        _iter_pi_tilde = {k: Var(sub_cons_ids, bounds=(0,None))}
        _iter_pi = {k: Var(sub_cons_ids, bounds=(0,None))}
        _iter_t = {k: Var(sub_cons_ids, bounds=(0,None))}
        _iter_lambda = {k: Var(sub_cons_ids, bounds=(0,None))}




        while k <= self._k_max_iter:

            # Step 2. Lower Bounding
            # Solve problem (P5) master problem.
            # This includes equations (53), (12), (13), and (15)
            # which is the highpoint relaxation (HPR)
            model_name = '_p5_alternate'
            lower_bounding_master = getattr(self._instance, model_name, None)
            if lower_bounding_master is None:
                xfrm = TransformationFactory('pao.bilevel.highpoint')
                kwds = {'submodel_name': model_name}
                xfrm.apply_to(self._instance, **kwds)
                lower_bounding_master = getattr(self._instance, model_name)

            # solve the HPR and check the termination condition
            lower_bounding_master.activate()
            with pyomo.opt.SolverFactory(solver) as opt:
                self.results.append(opt.solve(lower_bounding_master,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
            _check_termination_condition(self.results[-1])
            # get the objective function value for the HPR, which is the same as the main objective functi
            LB = [value(odata) for odata in self._instance.component_objects(Objective) if odata.parent_block() == self._instance][0]
            lower_bounding_master.deactivate()

            # Step 3. Termination
            if UB - LB < xi:
                return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))

            # fix the upper-level (master) variables to solve (P6) and (P7)
            for var in fixed_vars:
                var.fix(var.value)

            # Step 4. Subproblem 1
            # Solve problem (P6) lower-level problem for fixed upper-level optimal vars.
            with pyomo.opt.SolverFactory(solver) as opt:
                self.results.append(opt.solve(submodel,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
            _check_termination_condition(self.results[-1]) # check the last item appended to list
            theta = [value(odata) for odata in submodel.component_objects(Objective)][0]
            _fixed = array([var.value for (key, var) in matrix_repn._all_vars.items()])  # all variable values


            # Step 5. Subproblem 2
            # Solve problem (P7) upper bounding problem for fixed upper-level optimal vars.
            model_name = '_p7'
            upper_bounding_subproblem = getattr(self._instance, model_name, None)
            if upper_bounding_subproblem is None:
                xfrm = TransformationFactory('pao.bilevel.highpoint')
                kwds = {'submodel_name': model_name}
                xfrm.apply_to(self._instance, **kwds)
                upper_bounding_subproblem = getattr(self._instance, model_name)

                for odata in upper_bounding_subproblem.component_objects(Objective):
                    upper_bounding_subproblem.del_component(odata)

            # solve for the master problem objective value for just the lower level variables
            obj_constant = 0.
            obj_expr = 0.
            for var in c_vars+b_vars+i_vars:
                (C, C_q, C_constant) = matrix_repn.cost_vectors(self._instance, var)
                if obj_constant == 0. and C_constant != 0.:
                    obj_constant += C_constant # only add the constant once
                obj_expr += float(C + dot(C_q,_fixed))*var
            upper_bounding_subproblem.objective = Objective(expr=obj_expr + obj_constant)

            # include lower bound constraint on the subproblem objective
            sub_constant = 0.
            sub_expr = 0.
            for var in all_vars:
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                if sub_constant == 0. and C_constant != 0.:
                    sub_constant += C_constant # only add the constant once
                sub_expr += float(C + dot(C_q,_fixed))*var
            upper_bounding_subproblem.theta_pareto = Constraint(expr= sub_expr + sub_constant >= theta)

            with pyomo.opt.SolverFactory(solver) as opt:
                self.results.append(opt.solve(upper_bounding_subproblem,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
            _check_termination_condition(self.results[-1]) # check the last item appended to list

            # calculate new upper bound
            obj_constant = 0.
            obj_expr = 0.
            for var in fixed_vars:
                (C, C_q, C_constant) = matrix_repn.cost_vectors(self._instance, var)
                if obj_constant == 0. and C_constant != 0.:
                    obj_constant += C_constant # only add the constant once
                obj_expr += float(C + dot(C_q,_fixed))*var.value
            # line 16 of decomposition algorithm
            _ub = obj_expr + obj_constant + [value(odata) for odata in upper_bounding_subproblem.component_objects(Objective)][0]
            UB = min(UB,_ub)

            # unfix the submodel variables
            for var in fixed_vars:
                var.unfix()

            # Step 6. Tightening the Master Problem

            # constraint (74)
            lower_bounding_master.KKT_tight1 = Constraint(set(c_var_ids))
            for var in c_vars:
                _vid = id(var)
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                lhs_expr = float(C + dot(C_q,_fixed))*var
                rhs_expr = float(C + dot(C_q,_fixed))*_iter_c_tilde[k][_vid]
            lower_bounding_master.KKT_tight1[_vid] = lhs_expr >= rhs_expr

            # constraint (75a)
            lower_bounding_master.KKT_tight2a = Constraint(sub_cons_ids)
            lhs_expr_a = {key: 0. for key in sub_cons_ids}
            rhs_expr_a = {key: 0. for key in sub_cons_ids}
            for var in c_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    lhs_expr_a[key] += float(coef[_cid])*_iter_c_tilde[k][_vid]

            _vars = b_vars+i_vars+fixed_vars
            for var in _vars:
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    rhs_expr_a[key] += -float(coef[_cid])*var

            for key in sub_cons_ids:
                (id,sign) = key
                b = sub_cons_sense_rhs[key]
                rhs_expr_a[key] += b
                if sign=='l' or sign=='g->l':
                    lower_bounding_master.KKT_tight2a[key] = lhs_expr_a[key] <= rhs_expr_a[key]
                if sign=='e':
                    lower_bounding_master.KKT_tight2a[key] = lhs_expr_a[key] == rhs_expr_a[key]
                if sign=='g':
                    raise Exception('Problem encountered, this problem is not in standard form.')

            # constraint (75b)
            lower_bounding_master.KKT_tight2b = Constraint(sub_cons_ids)
            lhs_expr_b = {key: 0. for key in sub_cons_ids}
            rhs_expr_b = {key: 0. for key in sub_cons_ids}
            for var in c_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    lhs_expr_b[key] += float(coef[_cid])*_iter_pi_tilde[k][_vid]

                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                rhs_expr_b[key] = float(C + dot(C_q,_fixed))
            for key in sub_cons_ids:
                lower_bounding_master.KKT_tight2b[key] = lhs_expr_b[key] >= rhs_expr_b[key]

            # constraint (76a)
            lower_bounding_master.KKT_tight3a = ComplementarityList(sub_cons_ids)
            for key in sub_cons_ids:
                lower_bounding_master.KKT_tight3a[key] = complements(_iter_c_tilde[k][key] >= 0, lhs_expr_b[key] - rhs_expr_b[key] >= 0)

            # constraint (76b)
            lower_bounding_master.KKT_tight3b = ComplementarityList(sub_cons_ids)
            for key in sub_cons_ids:
                lower_bounding_master.KKT_tight3b[key] = complements(_iter_pi_tilde[k][key] >= 0, rhs_expr_a[key] - lhs_expr_a[key] >= 0)

            # constraint (77a)
            lower_bounding_master.KKT_tight4a = Constraint(c_var_ids)
            for key in c_var_ids:
                lower_bounding_master.KKT_tight4a[key] = _iter_c_tilde[k][key] >= 0

            # constraint (77b)
            lower_bounding_master.KKT_tight4a = Constraint(sub_cons_ids)
            for key in sub_cons_ids:
                lower_bounding_master.KKT_tight4b[key] = _iter_pi_tilde[k][key] >= 0

            # constraint (79)
            lower_bounding_master.projection1 = Constraint(sub_cons_ids)
            lhs_expr_proj = {key: 0. for key in sub_cons_ids}

            rhs_expr_proj = {key: 0. for key in sub_cons_ids}
            for var in c_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    lhs_expr_proj[key] += float(coef[_cid])*_iter_c[k][_vid]
                    #hs_expr_proj[key] -= _iter_t[k][_cid]

            for var in b_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    rhs_expr_proj[key] += -float(coef[_cid])*_iter_b[k][_vid]

            for var in i_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    rhs_expr_proj[key] += -float(coef[_cid])*_iter_i[k][_vid]

            for var in fixed_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    rhs_expr_proj[key] += -float(coef[_cid])*var

            for key in sub_cons_ids:
                (id,sign) = key
                b = sub_cons_sense_rhs[key]
                rhs_expr[key] += b
                if sign=='l' or sign=='g->l':
                    lower_bounding_master.projection1[key] = lhs_expr_proj[key] - _iter_t[k][key] <= rhs_expr_proj[key]
                if sign=='e':
                    lower_bounding_master.projection1[key] = lhs_expr_proj[key] - _iter_t[k][key] == rhs_expr_proj[key]
                if sign=='g':
                    raise Exception('Problem encountered, this problem is not in standard form.')

            # constraint (80a)
            lower_bounding_master.KKT_tight4a = Constraint(c_var_ids)
            for key in sub_cons_ids:
                lower_bounding_master.projection2a[key] = _iter_c[k][key] >= 0

            # constraint (80b)
            lower_bounding_master.KKT_tight4a = Constraint(sub_cons_ids)
            for key in sub_cons_ids:
                lower_bounding_master.projection2b[key] = _iter_t[k][key] >= 0

            disjunction = getattr(lower_bounding_master, 'disjunction', None)
            if disjunction is None:
                disjunction = Block(set(range(0,self._k_max_iter)))
            disjunction[k].indicator = Disjunct()
            disjunction[k].block = Disjunct()

            disjunction[k].indicator.cons = Constraint(sum(_iter_t[k][key] for key in sub_cons_ids) >= epsilon)

            # constraint (82a)
            lhs_constant = 0.
            lhs_expr = 0.
            rhs_constant = 0.
            rhs_expr = 0.
            for var in all_vars:
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                if lhs_constant == 0. and C_constant != 0.:
                    lhs_constant += C_constant # only add the constant once
                lhs_expr += float(C + dot(C_q,_fixed))*var
            for var in c_vars:
                _vid = id(var)
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                if lhs_constant == 0. and C_constant != 0.:
                    lhs_constant += C_constant # only add the constant once
                rhs_expr += float(C + dot(C_q,_fixed))*_iter_c[k][_vid]
            for var in b_vars:
                _vid = id(var)
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                if lhs_constant == 0. and C_constant != 0.:
                    lhs_constant += C_constant # only add the constant once
                rhs_expr += float(C + dot(C_q,_fixed))*_iter_b[k][_vid]
            for var in i_vars:
                _vid = id(var)
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                if lhs_constant == 0. and C_constant != 0.:
                    lhs_constant += C_constant # only add the constant once
                rhs_expr += float(C + dot(C_q,_fixed))*_iter_i[k][_vid]
            disjunction[k].block.projection3a = Constraint(expr= lhs_expr + lhs_constant >= rhs_expr + rhs_constant)

            # constraint (82b)
            disjunction[k].block.projection3b = Constraint(sub_cons_ids)
            for key in sub_cons_ids:
                (id,sign) = key
                if sign=='l' or sign=='g->l':
                    disjunction[k].block.projection3b[key] = lhs_expr_proj[key] <= rhs_expr_proj[key]
                if sign=='e':
                    disjunction[k].block.projection3b[key] = lhs_expr_proj[key] == rhs_expr_proj[key]
                if sign=='g':
                    raise Exception('Problem encountered, this problem is not in standard form.')


            # constraint (82c)
            disjunction[k].block.projection3c = Constraint(sub_cons_ids)
            lhs_expr = {key: 0. for key in sub_cons_ids}
            rhs_expr = {key: 0. for key in sub_cons_ids}
            for var in c_vars:
                _vid = id(var)
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for key in sub_cons_ids:
                    _cid = list(sub_cons_ids).index(key)
                    lhs_expr[key] += float(coef[_cid])*_iter_pi[k][_vid]

                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                rhs_expr[key] = float(C + dot(C_q,_fixed))
            for key in sub_cons_ids:
                lower_bounding_master.KKT_tight2b[key] = lhs_expr[key] >= rhs_expr[key]

            # constraint (82d)
            disjunction[k].block.projection3d = ComplementarityList(sub_cons_ids)
            for key in sub_cons_ids:
                disjunction[k].block.projection3d[key] = complements(_iter_c[k][key] >= 0, lhs_expr[key] - rhs_expr[key] >= 0)

            # constraint (82e)
            disjunction[k].block.projection3e = ComplementarityList(sub_cons_ids)
            for key in sub_cons_ids:
                disjunction[k].block.projection3e[key] = complements(_iter_pi[k][key] >= 0, rhs_expr_proj[key] - lhs_expr_proj[key] >= 0)

            # constraint (82f)
            disjunction[k].block.projection3f = Constraint(c_var_ids)
            for key in sub_cons_ids:
                disjunction[k].block.projection3f[key] = _iter_c[k][key] >= 0

            # constraint (82g)
            disjunction[k].block.projection3g = Constraint(sub_cons_ids)
            for key in sub_cons_ids:
                disjunction[k].block.projection3g[key] = _iter_pi[k][key] >= 0

            # need to do constraints 83-85

            # # set up the constraint sets needed
            # sub_con_set = set(key for key in matrix_repn._cons_sense_rhs[submodel.name].keys())
            # master_con_set = set(key for key in matrix_repn._cons_sense_rhs[self._instance.name].keys())
            # con_set = sub_con_set.union(master_con_set)
            # upper_bounding_subproblem.constraint = Constraint(con_set)
            #
            # # do the submodel constraints
            # con_expr_dict = matrix_repn._cons_sense_rhs[submodel.name]
            # con_range = range(0,len(con_expr_dict))
            # con_expr_list = [0. for i in con_range]
            # for var in i_vars:
            #     (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
            #     coef = A + dot(A_q.toarray(),_fixed)
            #
            #     for i in con_range:
            #         con_expr_list[i] += coef[i]*var
            #
            # for (k,s), rhs for con_expr_dict.items():
            #     idx = list(con_expr_dict).index((k,s))
            #     if s == 'l' or s == 'g->l':
            #         upper_bounding_subproblem.constraint[(k,s)] = con_expr_list[idx] <= rhs
            #     if s == 'e':
            #         upper_bounding_subproblem.constraint[(k,s)] = con_expr_list[idx] == rhs

            k = k + 1
            # Step 7. Loop
            if UB - LB < xi:
                return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))

    def _postsolve(self):

        # put problem back into original standard form
        for odata in self._instance.component_objects(Objective):
            if odata.parent_block() == self._instance:
                if self._upper_level_sense == maximize:
                    odata.set_value(-odata.expr)
                    odata.set_sense(maximize)
            if type(odata.parent_block()) == SubModel:
                if self._lower_level_sense == minimize:
                    odata.set_value(-odata.expr)
                    odata.set_sense(minimize)

