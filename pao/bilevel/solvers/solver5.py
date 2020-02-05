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
pao.bilevel.plugins.solver5

Declare the pao.bilevel.norbip solver.

"Near-Optimal Robust Bilevel Optimization"
M. Besancon, M.F. Anjos and L. Brotcorne
arXiv:1908.04040v5
Nov, 2019.

TODO: Currently handling linear subproblem; need to extend to
convex subproblem duality
"""

import time
import pyutilib.misc
import pyomo.opt
import pyomo.common
from pyomo.core import TransformationFactory, Var, Set, Block
from .solver3 import BilevelSolver3

safe_termination_conditions = [
    TerminationCondition.maxTimeLimit,
    TerminationCondition.maxIterations,
    TerminationCondition.minFunctionValue,
    TerminationCondition.minStepLength,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
    TerminationCondition.feasible,
    TerminationCondition.optimal,
    TerminationCondition.maxEvaluations,
    TerminationCondition.other,
]

@pyomo.opt.SolverFactory.register('pao.bilevel.norvep',
                                  doc='Solver for near-optimal vertex enumeration procedure')
class BilevelSolver5(pyomo.opt.OptSolver):
    """
    A solver that performs near-optimal robustness for bilevel programs
    """

    def __init__(self, **kwds):
        kwds['type'] = 'pao.bilevel.norvep'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()

        def _check_termination_condition(results):
            # do we want to be more restrictive of termination conditions?
            # do we want to have different behavior for sub-optimal termination?
            if results.solver.termination_condition not in safe_termination_conditions:
                raise Exception('Problem encountered during solve, termination_condition {}'.format(
                    results.solver.termination_condition))

        # construct the high-point problem (LL feasible, no LL objective)
        # s0 <- solve the high-point
        # if s0 infeasible then return high_point_infeasible
        xfrm = TransformationFactory('pao.bilevel.highpoint')
        xfrm.apply_to(self._instance)
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            solver = 'ipopt'

        for c in self._instance.component_objects(Block, descend_into=False):
            if '_hp' in c.name:
                c.activate()
                opt = SolverFactory(solver)
                results = opt.solve(c)
                _check_termination_condition(results)
                c.deactivate()

        # s1 <- solve the optimistic bilevel (linear/linear) problem (call solver3)
        # if s1 infeasible then return optimistic_infeasible
        opt = BilevelSolver3()
        opt.options.solver = solver
        results = opt.solve(self._instance)
        _check_termination_condition(results)
        # WIP: HIGHPOINT RELAXATION IS WORKING;
        # HOWEVER, WE NEED TO DEBUG ABOVE CODE BLOCK

        # for k \in [[m_u]] <for each constraint in the upper-level problem>
        # sk <- solve the dual adversarial  problem
        # if infeasible then return dual_adversarial_infeasible

        # Collect the vertices solutions for the dual adversarial problem
        # Solving the full problem sn0
        # Return the sn0 solution

    def _postsolve(self):
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
        #
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
        ##from pyomo.core import maximize
        ##if self._instance.sense == maximize:
            ##prob.sense = pyomo.opt.ProblemSense.maximize
        ##else:
            ##prob.sense = pyomo.opt.ProblemSense.minimize
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        #
        # Uncache the instance
        #
        self._instance = None
        return results
