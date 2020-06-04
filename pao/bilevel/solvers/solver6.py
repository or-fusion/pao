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
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.mpec import complements, ComplementarityList, Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pao.bilevel.solvers.solver_helpers import _check_termination_condition
from pao.bilevel.plugins.collect import BilevelMatrixRepn
from pao.bilevel.components import SubModel, varref, dataref
from pyomo.core import TransformationFactory, minimize, maximize, Block, Constraint, Objective, Var, Reals, Binary, Integers, Any
from pyomo.core.expr.numvalue import value
from numpy import array, dot
from pyomo.common.modeling import unique_component_name

@pyomo.opt.SolverFactory.register('pao.bilevel.ccg',
                                  doc='Solver for projection-based reformulation and decomposition.')
class BilevelSolver6(pyomo.opt.OptSolver):
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

    @property
    def _apply_solver(self):
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            solver = 'gurobi'

        bigm = TransformationFactory('gdp.bigm')

        # Step 1. Initialization
        LB = -inf
        UB = inf
        theta = 0.
        xi = 10e-3
        k = 0
        epsilon = 10e-3
        M = 1e6

        # matrix representation for bilevel problem
        matrix_repn = BilevelMatrixRepn(self._instance)

        # each lower-level problem
        submodel = [block for block in self._instance.component_objects(SubModel)][0]
        if len(submodel) != 1:
            raise Exception('Problem encountered, this is not a valid bilevel model for the solver.')
        self._instance.reclassify_component_type(submodel, Block)
        varref(submodel)
        dataref(submodel)

        # all algorithm blocks
        algorithm_blocks = list()
        algorithm_blocks.append(submodel)

        all_vars = {key: var for (key, var) in matrix_repn._all_vars.items()}
        # for k,v in all_vars.items():
        #     if v.ub is None:
        #         v.setub(M)
        #     if v.lb is None:
        #         v.setlb(-M)

        # get the variables that are fixed for the submodel (lower-level block)
        fixed_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._fixed_var_ids[submodel.name]}

        # continuous variables in SubModel
        c_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._c_var_ids - fixed_vars.keys()}

        # binary variables in SubModel
        b_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._b_var_ids - fixed_vars.keys()}

        # integer variables in SubModel
        i_vars = {key: var for (key, var) in matrix_repn._all_vars.items() if key in matrix_repn._i_var_ids - fixed_vars.keys()}

        # get constraint information related to constraint id, sign, and rhs value
        sub_cons = matrix_repn._cons_sense_rhs[submodel.name]

        while k <= self._k_max_iter:
            print('k={}'.format(k))
            # Step 2. Lower Bounding
            # Solve problem (P5) master problem.
            # This includes equations (53), (12), (13), (15), and (54)
            # On iteration k = 0, (54) does not exist. Instead of implementing (54),
            # this approach applies problem (P9) which incorporates KKT-based tightening
            # constraints, and a projection and indicator constraint set.
            model_name = '_p9'
            model_name = unique_component_name(self._instance, model_name)
            lower_bounding_master = getattr(self._instance, model_name, None)
            if lower_bounding_master is None:
                xfrm = TransformationFactory('pao.bilevel.highpoint')
                kwds = {'submodel_name': model_name}
                xfrm.apply_to(self._instance, **kwds)
                lower_bounding_master = getattr(self._instance, model_name)
                algorithm_blocks.append(lower_bounding_master)
                #self._instance.reclassify_component_type(submodel, Block)

                _c_var_bounds_rule = lambda m, k: c_vars[k].bounds
                _c_var_init_rule = lambda m, k: (c_vars[k].lb + c_vars[k].ub) / 2
                lower_bounding_master._iter_c = Var(Any, within=Reals, dense=False)  # set (iter k, c_var_ids)
                lower_bounding_master._iter_c_tilde = Var(c_vars.keys(), bounds=_c_var_bounds_rule, initialize=_c_var_init_rule,
                                                   within=Reals)  # set (iter k, c_var_ids)
                lower_bounding_master._iter_b = Var(Any, within=Binary, dense=False)  # set (iter k, b_var_ids)
                lower_bounding_master._iter_i = Var(Any, within=Integers, dense=False)  # set (iter k, i_var_ids)

                lower_bounding_master._iter_pi_tilde = Var(sub_cons.keys(), bounds=(0, M))  # set (iter k, sub_cons_ids)
                lower_bounding_master._iter_pi = Var(Any, bounds=(0, M), dense=False)  # set (iter k, sub_cons_ids)
                lower_bounding_master._iter_t = Var(Any, bounds=(0, M), dense=False)  # set (iter k, sub_cons_ids)
                lower_bounding_master._iter_lambda = Var(Any, bounds=(0, M), dense=False)  # set (iter k, sub_cons_ids)
                m = lower_bounding_master # shorthand reference to model

            # solution for all variable values
            _fixed = array([var.value for (key, var) in matrix_repn._all_vars.items()])

            if k == 0:
                # constraint (74)
                lhs_expr = 0.
                rhs_expr = 0.
                for _vid, var in c_vars.items():
                    (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                    coef = float(C)# + dot(C_q,_fixed))
                    ref = m._map[var]
                    lhs_expr += coef*ref
                    rhs_expr += coef*m._iter_c_tilde[_vid]
                expr = lhs_expr >= rhs_expr
                if not type(expr) is bool:
                   lower_bounding_master.KKT_tight1 = Constraint(expr=lhs_expr >= rhs_expr)

                # constraint (75a)
                lower_bounding_master.KKT_tight2a = Constraint(sub_cons.keys())
                lhs_expr_a = {key: 0. for key in sub_cons.keys()}
                rhs_expr_a = {key: 0. for key in sub_cons.keys()}
                for _vid, var in c_vars.items():
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                    coef = A #+ dot(A_q.toarray(), _fixed)
                    for _cid in sub_cons.keys():
                        idx = list(sub_cons.keys()).index(_cid)
                        lhs_expr_a[_cid] += float(coef[idx])*m._iter_c_tilde[_vid]

                for var in {**b_vars, **i_vars, **fixed_vars}.values():
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                    coef = A #+ dot(A_q.toarray(), _fixed)
                    for _cid in sub_cons.keys():
                        idx = list(sub_cons.keys()).index(_cid)
                        ref = m._map[var]
                        rhs_expr_a[_cid] += -float(coef[idx])*ref

                for _cid, b in sub_cons.items():
                    (_,sign) = _cid
                    rhs_expr_a[_cid] += b
                    if sign=='l' or sign=='g->l':
                        expr = lhs_expr_a[_cid] <= rhs_expr_a[_cid]
                    if sign=='e' or sign=='g':
                        raise Exception('Problem encountered, this problem is not in standard form.')
                    if not type(expr) is bool:
                        lower_bounding_master.KKT_tight2a[_cid] = expr
                    else:
                        lower_bounding_master.KKT_tight2a[_cid] = Constraint.Skip

                # constraint (75b)
                lower_bounding_master.KKT_tight2b = Constraint(c_vars.keys())
                lhs_expr_b = {key: 0. for key in c_vars.keys()}
                rhs_expr_b = {key: 0. for key in c_vars.keys()}
                for _vid, var in c_vars.items():
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                    coef = A #+ dot(A_q.toarray(), _fixed)
                    lhs_expr_b[_vid] += float(sum(coef))*m._iter_pi_tilde[_vid]

                    (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                    rhs_expr_b[_vid] = float(C) #+ dot(C_q,_fixed))

                    expr = lhs_expr_b[_vid] >= rhs_expr_b[_vid]
                    if not type(expr) is bool:
                        lower_bounding_master.KKT_tight2b[_vid] = expr
                    else:
                        lower_bounding_master.KKT_tight2b[_vid] = Constraint.Skip

                # constraint (76a)
                lower_bounding_master.KKT_tight3a = ComplementarityList()
                for _vid in c_vars.keys():
                    lower_bounding_master.KKT_tight3a.add(complements(m._iter_c_tilde[_vid] >= 0, lhs_expr_b[_vid] - rhs_expr_b[_vid] >= 0))

                # constraint (76b)
                lower_bounding_master.KKT_tight3b = ComplementarityList()
                for _cid in sub_cons.keys():
                    lower_bounding_master.KKT_tight3b.add(complements(m._iter_pi_tilde[_cid] >= 0, rhs_expr_a[_cid]  - lhs_expr_a[_cid] >= 0))

                # constraint (77a)
                lower_bounding_master.KKT_tight4a = Constraint(c_vars.keys())
                for _vid in c_vars.keys():
                    lower_bounding_master.KKT_tight4a[_vid] = m._iter_c_tilde[_vid] >= 0

                # constraint (77b)
                lower_bounding_master.KKT_tight4b = Constraint(sub_cons.keys())
                for _cid in sub_cons.keys():
                    lower_bounding_master.KKT_tight4b[_cid] = m._iter_pi_tilde[_cid] >= 0

            # solve the HPR and check the termination condition
            lower_bounding_master.activate()
            TransformationFactory('mpec.simple_disjunction').apply_to(lower_bounding_master)
            bigm.apply_to(self._instance, targets=lower_bounding_master)

            with pyomo.opt.SolverFactory(solver) as opt:
                self.results.append(opt.solve(lower_bounding_master,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
            _check_termination_condition(self.results[-1])
            # the LB should be a sequence of non-decreasing lower-bounds
            _lb = [value(odata) for odata in self._instance.component_objects(Objective) if odata.parent_block() == self._instance][0]
            if _lb < LB:
                raise Exception('The lower-bound should be non-decreasing; a decreasing lower-bound indicates an algorithm issue.')
            LB = max(LB,_lb)
            lower_bounding_master.deactivate()

            # Step 3. Termination
            if UB - LB < xi:
                print(UB)
                print(LB)
                return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))

            # fix the upper-level (master) variables to solve (P6) and (P7)
            for key, var in fixed_vars.items():
                var.fix(var.value)

            # Step 4. Subproblem 1
            # Solve problem (P6) lower-level problem for fixed upper-level optimal vars.
            # In iteration k=0, this first subproblem is always feasible; furthermore, the
            # optimal solution to (P5), alternatively (P9), will also be feasible to (P6).
            with pyomo.opt.SolverFactory(solver) as opt:
                self.results.append(opt.solve(submodel,
                                              tee=self._tee,
                                              timelimit=self._timelimit))
            _check_termination_condition(self.results[-1]) # check the last item appended to list
            theta = [value(odata) for odata in submodel.component_objects(Objective)][0]

            # solution for all variable values
            _fixed = array([var.value for (key, var) in matrix_repn._all_vars.items()])


            # Step 5. Subproblem 2
            # Solve problem (P7) upper bounding problem for fixed upper-level optimal vars.
            model_name = '_p7'
            model_name = unique_component_name(self._instance, model_name)
            upper_bounding_subproblem = getattr(self._instance, model_name, None)
            if upper_bounding_subproblem is None:
                xfrm = TransformationFactory('pao.bilevel.highpoint')
                kwds = {'submodel_name': model_name}
                xfrm.apply_to(self._instance, **kwds)
                upper_bounding_subproblem = getattr(self._instance, model_name)
                algorithm_blocks.append(upper_bounding_subproblem)
                #self._instance.reclassify_component_type(submodel, Block)

                for odata in upper_bounding_subproblem.component_objects(Objective):
                    upper_bounding_subproblem.del_component(odata)

            # solve for the master problem objective value for just the lower level variables
            upper_bounding_subproblem.del_component('objective')
            obj_constant = 0.
            obj_expr = 0.
            for var in {**c_vars, **b_vars, **i_vars}.values():
                (C, C_q, C_constant) = matrix_repn.cost_vectors(self._instance, var)
                if obj_constant == 0. and C_constant != 0.:
                    obj_constant += C_constant # only add the constant once
                obj_expr += float(C + dot(C_q,_fixed))*var
            upper_bounding_subproblem.objective = Objective(expr=obj_expr + obj_constant)

            # include lower bound constraint on the subproblem objective
            upper_bounding_subproblem.del_component('theta_pareto')
            sub_constant = 0.
            sub_expr = 0.
            for var in all_vars.values():
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
            for var in fixed_vars.values():
                (C, C_q, C_constant) = matrix_repn.cost_vectors(self._instance, var)
                if obj_constant == 0. and C_constant != 0.:
                    obj_constant += C_constant # only add the constant once
                obj_expr += float(C + dot(C_q,_fixed))*var.value
            # line 16 of decomposition algorithm
            _ub = obj_expr + obj_constant + [value(odata) for odata in upper_bounding_subproblem.component_objects(Objective)][0]
            UB = min(UB,_ub)

            # unfix the upper-level variables
            for var in fixed_vars.values():
                var.unfix()

            # fix the solution for submodel binary variables
            for _vid, var in b_vars.items():
                m._iter_b[(k, _vid)].fix(var.value)

            # fix the solution for submodel integer variables
            for _vid, var in i_vars.items():
                m._iter_i[(k, _vid)].fix(var.value)

            # Step 6. Tightening the Master Problem
            projections = getattr(lower_bounding_master, 'projections', None)
            if projections is None:
                lower_bounding_master.projections = Block(Any)
                projections = lower_bounding_master.projections

            # constraint (79)
            projections[k].projection1 = Constraint(sub_cons.keys())
            lhs_expr_proj = {key: 0. for key in sub_cons.keys()}
            rhs_expr_proj = {key: 0. for key in sub_cons.keys()}
            for _vid, var in c_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for _cid in sub_cons.keys():
                    idx = list(sub_cons.keys()).index(_cid)
                    lhs_expr_proj[_cid] += float(coef[idx])*m._iter_c[(k,_vid)]

            for _vid, var in b_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for _cid in sub_cons.keys():
                    idx = list(sub_cons.keys()).index(_cid)
                    rhs_expr_proj[_cid] += -float(coef[idx])*m._iter_b[(k,_vid)]

            for _vid, var in i_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for _cid in sub_cons.keys():
                    idx = list(sub_cons.keys()).index(_cid)
                    rhs_expr_proj[_cid] += -float(coef[idx])*m._iter_i[(k,_vid)]

            for _vid, var in fixed_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for _cid in sub_cons.keys():
                    idx = list(sub_cons.keys()).index(_cid)
                    rhs_expr_proj[_cid] += -float(coef[idx])*var

            for _cid, b in sub_cons.items():
                (id,sign) = _cid
                rhs_expr_proj[_cid] += b
                if sign=='l' or sign=='g->l':
                    projections[k].projection1[_cid] = lhs_expr_proj[_cid] - m._iter_t[(k,_cid)] <= rhs_expr_proj[_cid]
                if sign=='e' or sign=='g':
                    raise Exception('Problem encountered, this problem is not in standard form.')

            # constraint (80a)
            projections[k].projection2a = Constraint(sub_cons.keys())
            for _vid in c_vars.keys():
                projections[k].projection2a[_vid] = m._iter_c[(k,_vid)] >= 0

            # constraint (80b)
            projections[k].projection2b = Constraint(sub_cons.keys())
            for _cid in sub_cons.keys():
                projections[k].projection2b[_cid] = m._iter_t[(k,_cid)] >= 0

            # constraint (82)
            projections[k].projection3 = Block()
            projections[k].projection3.indicator = Disjunct()
            projections[k].projection3.block = Disjunct()
            projections[k].projection3.disjunct = Disjunction(expr=[projections[k].projection3.indicator,projections[k].projection3.block])

            projections[k].projection3.indicator.cons_feas = Constraint(expr=sum(m._iter_t[(k,_cid)] for _cid in sub_cons.keys()) >= epsilon)

            disjunction = projections[k].projection3.block

            # constraint (82a)
            lhs_constant = 0.
            lhs_expr = 0.
            rhs_constant = 0.
            rhs_expr = 0.
            for _vid, var in {**c_vars,**b_vars,**i_vars}.items():
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                lhs_expr += float(C + dot(C_q,_fixed))*var
                if var.is_continuous():
                    rhs_expr += float(C + dot(C_q, _fixed)) * m._iter_c[(k,_vid)]
                if var.is_binary():
                    rhs_expr += float(C + dot(C_q, _fixed)) * m._iter_b[(k,_vid)]
                if var.is_integer():
                    rhs_expr += float(C + dot(C_q, _fixed)) * m._iter_i[(k,_vid)]
            disjunction.projection3a = Constraint(expr= lhs_expr + lhs_constant >= rhs_expr + rhs_constant)

            # constraint (82b)
            disjunction.projection3b = Constraint(sub_cons.keys())
            for _cid in sub_cons.keys():
                (_,sign) = _cid
                if sign=='l' or sign=='g->l':
                    expr = lhs_expr_proj[_cid] <= rhs_expr_proj[_cid]
                if sign=='e' or sign=='g':
                    raise Exception('Problem encountered, this problem is not in standard form.')
                if not type(expr) is bool:
                    disjunction.projection3b[_cid] = expr
                else:
                    disjunction.projection3b[_cid] = Constraint.Skip

            # constraint (82c)
            disjunction.projection3c = Constraint(c_vars.keys())
            lhs_expr = {key: 0. for key in c_vars.keys()}
            rhs_expr = {key: 0. for key in c_vars.keys()}
            for _vid, var in c_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                lhs_expr[_vid] += float(sum(coef))*m._iter_pi_tilde[_vid]

                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                rhs_expr[_vid] = float(C + dot(C_q,_fixed))

                expr = lhs_expr[_vid] >= rhs_expr[_vid]
                if not type(expr) is bool:
                    disjunction.projection3c[_vid] = expr
                else:
                    disjunction.projection3c[_vid] = Constraint.Skip

            # constraint (82d)
            disjunction.projection3d = ComplementarityList()
            for _vid in c_vars.keys():
                disjunction.projection3d.add(complements(m._iter_c[(k,_vid)] >= 0, lhs_expr[_vid] - rhs_expr[_vid] >= 0))

            # constraint (82e)
            disjunction.projection3e = ComplementarityList()
            for _cid in sub_cons.keys():
                disjunction.projection3e.add(complements(m._iter_pi[(k,_cid)] >= 0, rhs_expr_proj[_cid] - lhs_expr_proj[_cid] >= 0))

            # constraint (82f)
            disjunction.projection3f = Constraint(c_vars.keys())
            for _vid in c_vars.keys():
                disjunction.projection3f[_cid] = m._iter_c[(k,_vid)] >= 0

            # constraint (82g)
            disjunction.projection3g = Constraint(sub_cons.keys())
            for _cid in sub_cons.keys():
                disjunction.projection3g[_cid] = m._iter_pi[(k,_cid)] >= 0

            # constraint (83a)
            projections[k].projection4a = Constraint(c_vars.keys())
            lhs_expr = {key: 0. for key in c_vars.keys()}
            for _vid, var in c_vars.items():
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                coef = A + dot(A_q.toarray(), _fixed)
                for _cid in sub_cons.keys():
                    idx = list(sub_cons.keys()).index(_cid)
                    lhs_expr[_vid] += float(coef[idx])*m._iter_lambda[(k,_cid)]

                expr = lhs_expr[_vid] >= 0
                if not type(expr) is bool:
                    projections[k].projection4a[_vid] = expr
                else:
                    projections[k].projection4a[_vid] = Constraint.Skip

            # constraint (83b)
            projections[k].projection4b = ComplementarityList()
            for _vid in c_vars.keys():
                projections[k].projection4b.add(complements(m._iter_c[(k,_vid)] >= 0, lhs_expr[_vid] >= 0))

            # constraint (84a)
            projections[k].projection5a = Constraint(sub_cons.keys())
            for _cid in sub_cons.keys():
                projections[k].projection5a[_cid] = 1-m._iter_lambda[(k,_cid)] >= 0

            # constraint (84b)
            projections[k].projection5b = ComplementarityList()
            for _cid in sub_cons.keys():
                projections[k].projection5b.add(complements(m._iter_t[(k,_cid)] >= 0, 1 - m._iter_lambda[(k,_cid)] >= 0))

            # constraint (85)
            projections[k].projection6 = ComplementarityList()
            for _cid in sub_cons.keys():
                projections[k].projection6.add(complements(m._iter_lambda[(k,_cid)] >= 0, rhs_expr_proj[_cid] - lhs_expr_proj[_cid] + m._iter_t[(k,_cid)] >= 0))

            # Transform all the complementarity to be MILP representable
            #TransformationFactory('mpec.simple_disjunction').apply_to(projections[k].projection3.block)
            #TransformationFactory('mpec.simple_disjunction').apply_to(projections[k])
            #TransformationFactory('mpec.simple_disjunction').apply_to(projections[k])

            k = k + 1
            # Step 7. Loop
            if UB - LB < xi:
                print(UB)
                print(LB)
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

