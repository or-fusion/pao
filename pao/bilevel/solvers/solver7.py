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
pao.bilevel.plugins.solver7

Declare the ld solver for a stochastic bilevel problem
"""

import time
import pyutilib.misc
from pyomo.core.base.block import _BlockData
from pyomo.core.base.constraint import IndexedConstraint
from pyomo.core import TransformationFactory, Var, Constraint, Block, Objective, Set
import pyomo.opt
import pyomo.common
from itertools import chain
from pao.bilevel.solvers.solver_utils import safe_termination_conditions

@pyomo.opt.SolverFactory.register('pao.bilevel.stochastic_ld',
                                  doc=\
'Solver for stochastic bilevel interdiction problems using linear duality')
class BilevelSolver7(pyomo.opt.OptSolver):
    """
    A solver that optimizes a bilevel program interdiction problem, where
    (1) the upper objective is the opposite sense of the lower objectives
    and is the weighted sum of these objectives, and
    (2) the lower problems are linear and continuous.
    """

    def __init__(self, **kwds):
        kwds['type'] = 'pao.bilevel.stochastic_ld'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        # TODO: Override _presolve to ensure that we are passing
        #   all options to the solver (e.g., the io_options)
        self.resolve_subproblem = True #kwds.pop('resolve_subproblem', True)
        self.use_dual_objective = True #kwds.pop('use_dual_objective', True)
        self.subproblem_objective_weights = kwds.pop('subproblem_objective_weights', None)
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()
        #
        # Cache the instance
        #
        if self.subproblem_objective_weights is None:
            raise Exception('Problem encountered, expected probability weights.')

        xfrm = TransformationFactory('pao.bilevel.linear_dual')
        xfrm.apply_to(self._instance, use_dual_objective=self.use_dual_objective, \
                      subproblem_objective_weights=self.subproblem_objective_weights)
        #
        # Verify whether the model is linear
        #
        nonlinear = False
        for odata in chain(self._instance.component_objects(Objective, active=True), \
                           self._instance.component_objects(Constraint, active=True, descend_into=True)):
            if type(odata) == IndexedConstraint:
                for _name, _odata in odata.items():
                    nonlinear = _odata.expr.polynomial_degree() != 1
                    if nonlinear:
                        break
            else:
                nonlinear = odata.expr.polynomial_degree() != 1
            if nonlinear:
                # Stop after the first occurrence in the objective or one of the constraints
                break
        #
        # Apply an additional transformation to remap bilinear terms
        #
        if nonlinear:
            gdp_xfrm = TransformationFactory("gdp.bilinear")
            gdp_xfrm.apply_to(self._instance)
            mip_xfrm = TransformationFactory("gdp.bigm")
            mip_xfrm.apply_to(self._instance, bigM=self.options.get('bigM', 10))
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:     #pragma:nocover
            solver = 'glpk'
        #
        with pyomo.opt.SolverFactory(solver) as opt:
            self.results = []
            #
            #
            #opt.options['mipgap'] = self.options.get('mipgap', 0.001)
            results = opt.solve(self._instance,
                                          tee=self._tee,
                                          timelimit=self._timelimit)
            self.results.append(results)

            if results.solver.termination_condition not in safe_termination_conditions:
                raise Exception('Problem encountered during solve, termination_condition {}'.format(
                    results.solver.termination_condition))

            #
            # If the problem was bilinear, then reactivate the original data
            #
            if nonlinear:
                i = 0
                for v in self._instance.bilinear_data_.vlist.itervalues():
                    if abs(v.value) <= 1e-7:
                        self._instance.bilinear_data_.vlist_boolean[i] = 0
                    else:
                        self._instance.bilinear_data_.vlist_boolean[i] = 1
                    i = i + 1
                #
                self._instance.bilinear_data_.deactivate()
            if self.resolve_subproblem:
                #
                # Transform the result back into the original model
                #
                tdata = self._instance._transformation_data['pao.bilevel.linear_dual']
                unfixed_tdata = list()
                # Copy variable values and fix them
                for v in tdata.fixed:
                    if not v.fixed:
                        if v.is_binary():
                            v.value = round(self._instance.find_component(v).value)
                        if v.is_integer():
                            v.value = round(self._instance.find_component(v).value)
                        else:
                            v.value = self._instance.find_component(v).value
                        v.fixed = True
                        unfixed_tdata.append(v)
                # Reclassify the SubModel components and resolve
                for name_ in tdata.submodel:
                    submodel = self._instance.find_component(name_)
                    submodel.activate()
                    for data in submodel.component_map(active=False).values():
                        if not isinstance(data, Var) and not isinstance(data, Set) and not isinstance(data, Objective):
                            data.activate()
                    _dual_name = name_+'_dual'
                    _parent = submodel.parent_block()
                    if type(_parent) == _BlockData:
                        _dual_name = _dual_name.replace(_parent.name + ".", "")
                        dual_submodel = getattr(_parent,_dual_name)
                        dual_submodel.deactivate()
                        pyomo.common.PyomoAPIFactory('pyomo.repn.compute_standard_repn')({}, model=submodel)
                        _submodel_name = name_.replace(_parent.name + ".", "")
                        _parent.reclassify_component_type(_submodel_name, Block)
                    else:
                        dual_submodel = self._instance.find_component(_dual_name)
                        dual_submodel.deactivate()
                        pyomo.common.PyomoAPIFactory('pyomo.repn.compute_standard_repn')({}, model=submodel)
                        self._instance.reclassify_component_type(name_, Block)
                    #
                    # Use the with block here so that deactivation of the
                    # solver plugin always occurs thereby avoiding memory
                    # leaks caused by plugins!
                    #
                if self.use_dual_objective:
                    for data in self._instance.component_map(active=False).values():
                        if isinstance(data, Objective):
                            data.activate()
                            data.sense = - data.sense

                with pyomo.opt.SolverFactory(solver) as opt_inner:
                    #
                    # **NOTE: It would be better to override _presolve on the
                    #         base class of this solver as you might be
                    #         missing a number of keywords that were passed
                    #         into the solve method (e.g., none of the
                    #         io_options are getting relayed to the subsolver
                    #         here).
                    #
                    #opt_inner.options['mipgap'] = self.options.get('mipgap', 0.001)
                    results = opt_inner.solve(self._instance,
                                              tee=self._tee,
                                              timelimit=self._timelimit)
                    self.results.append(results)

                # Unfix variables
                for v in tdata.fixed:
                    if v in unfixed_tdata:
                        v.fixed = False

                # revert objective sense
                if self.use_dual_objective:
                    for data in self._instance.component_map(active=False).values():
                        if isinstance(data, Objective):
                            data.sense = - data.sense

            # check that the solutions list is not empty
            if self._instance.solutions.solutions:
                self._instance.solutions.select(0, ignore_fixed_vars=True)
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            self.results_obj = self._setup_results_obj()
            #
            # Reactivate top level objective
            #
            for odata in self._instance.component_map(Objective).values():
                odata.activate()
            #
            # Return the sub-solver return condition value and log
            #
            return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                       log=getattr(opt, '_log', None))

    def _postsolve(self):
        #
        # Uncache the instance
        #
        self._instance = None
        #
        # Return the results object
        #
        return self.results_obj

    def _setup_results_obj(self):
        #
        # Create a results object
        #
        results = pyomo.opt.SolverResults()
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        cpu_ = []
        for res in self.results:
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append(res.solver.cpu_time)
        if cpu_:
            solv.cpu_time = sum(cpu_)
        #
        # TODO: detect infeasibilities, etc
        solv.termination_condition = pyomo.opt.TerminationCondition.optimal
        #
        # PROBLEM
        #
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables =\
            self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables =\
            self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        return results
