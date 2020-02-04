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
from pyomo.core import TransformationFactory, Var, Set


@pyomo.opt.SolverFactory.register('pao.bilevel.norvep',
                                  doc='Solver for near-optimal vertex enumeration procedure')
class BilevelSolver5(pyomo.opt.OptSolver):
    """
    A solver that performs global optimization of bilevel
    quadratic programs.
    """

    def __init__(self, **kwds):
        kwds['type'] = 'pao.bilevel.bqp'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds): pass
        # self._instance = args[0]
        # pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self): pass



        # construct the high-point problem (LL feasible, no LL objective)
        # s0 <- solve the high-point
        # if s0 infeasible then return high_point_infeasible

        # s1 <- solve the optimistic bilevel (linear/linear) problem (call solver3)
        # if s1 infeasible then return optimistic_infeasible

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
