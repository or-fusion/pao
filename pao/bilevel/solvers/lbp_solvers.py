#
# Solvers that convert the problem to a LinearBilevelProblem and
# solve using a LBP-specific solver.
#
import time
#import pyutilib.misc
#from pyomo.core.base.block import _BlockData
#from pyomo.core.base.constraint import IndexedConstraint
#from pyomo.core import TransformationFactory, Var, Constraint, Block, Objective, Set
import pyomo.opt
#import pyomo.common
#from itertools import chain
#from pao.bilevel.solvers.solver_utils import safe_termination_conditions
import pao.lbp


class LBPSolverBase(pyomo.opt.OptSolver):

    def __init__(self, lbp_solver, **kwds):
        super(pyomo.opt.OptSolver, self).__init__(**kwds)
        self._lbp_solver = lbp_solver
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        # TODO: Override _presolve to ensure that we are passing
        #   all options to the solver (e.g., the io_options)
        #self.resolve_subproblem = True #kwds.pop('resolve_subproblem', True)
        #self.use_dual_objective = True #kwds.pop('use_dual_objective', True)
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _postsolve(self):
        #
        # Uncache the instance
        #
        self._instance = None
        #
        # Return the results object
        #
        return self.results_obj

    def _setup_results_obj(self, lbp_results):
        #
        # Create a results object
        #
        results = pyomo.opt.SolverResults()
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self._lbp_solver
        solv.subsolver = self.options.subsolver
        solv.wallclock_time = self.wall_time
        # TODO - translate to Pyomo termination condition
        solv.termination_condition = lbp_results.termination_condition
        #
        # PROBLEM
        #
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables = self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables = self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        return results

    def _apply_solver(self):
        start_time = time.time()
        #
        # Cache the instance
        #
        try:
            lbp, soln_manager = convert_pyomo2LinearBilevelProblem(self._instance)
        except RuntimeError as err:
            print("Cannot convert Pyomo model to a LinearBilevelProblem")
            raise
        #
        with pao.lbp.LinearBilevelSolver(self._lbp_solver) as opt:
            results = opt.solve(lbp, options=self.options, tee=self._tee, timelimit=self._timelimit)

            soln_manager.copy_from_to(lbp, self._instance)
            
            self.wall_time = time.time() - start_time
            self.results_obj = self._initialize_results_obj(results)
        #
        # Return the sub-solver return condition value and log
        #
        return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                   log=getattr(opt, '_log', None))


@pyomo.opt.SolverFactory.register('pao.bilevel.lbp_FA',
                                  doc=pao.lbp.LinearBilevelSolver.doc('pao.lbp.FA'))
class LBP_Solver1(LBPSolverBase):

    def __init__(self, **kwds):
        super(LBPSolverBase, self).__init__('pao.bilevel.lbp_FA', **kwds)

