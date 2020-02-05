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
import pyutilib.misc
import pyomo.opt
from math import inf
import pyomo.common
from pyomo.core import TransformationFactory, Var, Set, Block

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
        self._k_max_iter = 100

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        # Step 1. Initialization
        LB = -inf
        UB = inf
        xi = 0
        k = 0
        YL0 = None

        while k <= self._k_max_iter:
            # Step 2. Lower Bounding
            # Solve problem (P5) master problem.
            solver = self.options.solver
            if not self.options.solver:
                solver = 'gurobi'

            opt = SolverFactory(solver)
            # Step 3. Termination
            if UB - LB < xi:
                return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))

            # Step 4. Subproblem 1
            # Solve problem (P6) lower-level problem for fixed upper-level optimal vars.

            # Step 5. Subproblem 2
            # Solve problem (P7) upper bounding problem.

            # Step 6. Tightening the Master Problem

            k = k + 1
            # Step 7. Loop
            if UB - LB < xi:
                return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))

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
