#
# A solver for linear bilevel programs using
# using projected column constraint generation
# "A projection-based reformulation and decomposition algorithm for global optimization 
#  of a class of mixed integer bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You
#
# Adapted from an implementation by She'ifa Punla-Green at Sandia National Labs
#
#This algorithm seeks to solve the following bilevel MILP:
#    min cR*xu + cZ*yu + dR*xl0 + dZ*yl0 
#    s.t. AR*xu + AZ*yu + BR*xl0 + BZ* yl0 <= r
#     (xl0,yl0) in argmax {wR*xl+wZ*yl: PR*xl+PZ*yl<=s-QR*xu-QZ*yu}
#
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.mpec import ComplementarityList, complements
from ..solver import LinearBilevelSolver, LinearBilevelSolverBase, LinearBilevelResults
from ..repn import LinearBilevelProblem
from ..convert_repn import convert_LinearBilevelProblem_to_standard_form
from .. import pyomo_util


@LinearBilevelSolver.register(
        name='pao.lbp.PCCG',
        doc='A solver for linear bilevel programs using using projected column constraint generation')
class LinearBilevelSolver_FA(LinearBilevelSolverBase):

    def __init__(self, **kwds):
        super(LinearBilevelSolverBase, self).__init__(name='pao.lbp.PCCG')
        self.config.solver = 'glpk'

    def check_model(self, lbp):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(lbp) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.name
        lbp.check()
        #
        assert (len(lbp.L) == 1), "Can only solve linear bilevel problems with one lower-level"
        #
        # No binary or integer lower level variables
        #
        #for i in range(len(lbp.L)):
        #    assert (len(lbp.L[i].xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.name
        #    assert (len(lbp.L[i].xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.name

    def solve(self, lbp, options=None, **config_options):
        #
        # Error checks
        #
        self.check_model(lbp)
        #
        # Process keyword options
        #
        self._update_config(config_options)
        #
        # Start clock
        #
        start_time = time.time()

        # TODO - does the standard form need to use inequalities?
        self.standard_form, soln_manager = convert_LinearBilevelProblem_to_standard_form(lbp)
        
        results = LinearBilevelResults(solution_manager=soln_manager)

        execute_PCCG_solver(M, self.config, results)

        results.solver.wallclock_time = time.time() - start_time
        return results

    def _initialize_results(self, results, pyomo_results, M):
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.config.solver
        solv.termination_condition = pyomo_results.solver.termination_condition
        solv.solver_time = pyomo_results.solver.time
        if self.config.load_solutions:
            solv.best_feasible_objective = pe.value(M.o)
        #
        # PROBLEM - Maybe this should be the summary of the BLP itself?
        #
        prob = results.problem
        prob.name = M.name
        prob.number_of_objectives = pyomo_results.problem.Number_of_objectives
        prob.number_of_constraints = pyomo_results.problem.Number_of_constraints
        prob.number_of_variables = pyomo_results.problem.Number_of_variables
        prob.number_of_nonzeros = pyomo_results.problem.Number_of_nonzeros
        prob.lower_bound = pyomo_results.problem.Lower_bound
        prob.upper_bound = pyomo_results.problem.Upper_bound
        prob.sense = 'minimize'
        return results

    def _debug(self):
        for j in M.U.xR:
            print("U",j,pe.value(M.U.xR[j]))
        for j in M.L.xR:
            print("L",j,pe.value(M.L.xR[j]))
        for j in M.kkt.lam:
            print("lam",j,pe.value(M.kkt.lam[j]))
        for j in M.kkt.nu:
            print("nu",j,pe.value(M.kkt.nu[j]))

