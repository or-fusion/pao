#
# A solver for linear bilevel problems that
# represent interdiction problems where the upper-
# and lower-objectives are opposite.
#
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt

import pao.common
from ..solver import Solver, LinearMultilevelSolverBase, LinearMultilevelResults
from ..repn import LinearMultilevelProblem
from ..convert_repn import convert_to_standard_form
from . import pyomo_util

nan = float('nan')


@Solver.register(
        name="pao.mpr.interdiction",
        doc='PAO solver for Multilevel Problem Representations that define linear interdiction problems, where the upper- and lower-objectives are opposite.')
class LinearMultilevelSolver_interdiction(LinearMultilevelSolverBase):
    """
    PAO LD solver for linear interdiction MPRs: pao.mpr.LD

    This solver replaces lower-level problems using the dual and
    calls a MIP solver to solve the reformulated problem.
    """

    def __init__(self, **kwds):
        super().__init__(name='pao.mpr.interdiction')
        self.config.solver = 'glpk'
        self.config.mipgap = nan

    def check_model(self, mpr):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(mpr) is LinearMultilevelProblem), "Solver '%s' can only solve a LinearMultilevelProblem" % self.solver_type
        mpr.check()
        #
        # TODO: For now, we just deal with the case where there is a single lower-level.  Later, we
        # will generalize this.
        #
        assert (len(mpr.L) == 1), "Only one lower-level is handled right now"
        #
        # No binary or integer lower level variables
        #
        for i in range(len(mpr.L)):
            assert (len(mpr.L[i].xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
            assert (len(mpr.L[i].xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type
        #
        # Upper and lower objectives are the opposite of each other
        #
        for i in range(len(mpr.L)):
            assert (mpr.check_opposite_objectives(mpr.U, mpr.L[i])), "Lower level L[%d] does not have an objective that is the opposite of the upper-level" % i
        #
        # Lower level variables are not allowed in the upper-level
        # constraints.
        #
        # TODO - confirm that this is a necessarily restriction for this
        # solver.
        #
        for i in range(len(mpr.L)):
            assert (U.A.L[i].xR is None), "The lower-level variables cannot be used in the upper-level constraints."
        #
        # Upper-level variables are not allowed in the lower-level
        # constraints.
        #
        # NOTE: this avoids the introduction of quadratic terms when
        # dualizing the lower-level.  That might be OK, but for now we
        # are assuming the simple application of a MIP solver.
        #
        for i in range(len(mpr.L)):
            assert (L[i].A.U.xR is None), "The upper-level variables cannot be used in the lower-level constraints."
            assert (L[i].A.U.xZ is None), "The upper-level variables cannot be used in the lower-level constraints."
            assert (L[i].A.U.xB is None), "The upper-level variables cannot be used in the lower-level constraints."

    def solve(self, mpr, options=None, **config_options):
        #
        # Error checks
        #
        self.check_model(mpr)
        #
        # Process keyword options
        #
        self._update_config(config_options)
        #
        # Start clock
        #
        start_time = time.time()

        self.standard_form, soln_manager = convert_to_standard_form(mpr)

        M = self._create_pyomo_model(self.standard_form)
        #
        # Solve the Pyomo model the specified solver
        #
        with pyomo.opt.SolverFactory(self.config.solver) as opt:
            if self.config.mipgap is not nan:
                opt.options['mipgap'] = self.config.mipgap
            if options is not None:
                opt.options.update(options)
            pyomo_results = opt.solve(M, tee=tee, 
                                         load_solutions=self.config.load_solutions)
            pyomo.opt.check_termination(pyomo_results)

            self._initialize_results(results, pyomo_results, M)
            results.solver.rc = getattr(opt, '_rc', None)

            if self.config.load_solutions:
                # Load results from the Pyomo model to the LinearMultilevelProblem
                results.copy_solution(From=M, To=mpr)
            else:
                # Load results from the Pyomo model to the Results
                results.load_from(pyomo_results)

            #self._debug()
            #results.solver.log = getattr(opt, '_log', None)

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

    def _create_pyomo_model(self, repn):
        #
        # Create Pyomo model
        #
        M = pe.ConcreteModel()
        M.U = pe.Block()
        M.L = pe.Block()
        M.dual = pe.Block()

        # upper- and lower-level variables
        pyomo_util._create_variables(repn.U, M.U)
        pyomo_util._create_variables(repn.L, M.L)
        M.dual.z = pe.Var()

        # upper- and lower-level constraints
        pyomo_util.add_linear_constraints(M.U, repn.U.A, repn.U, repn.L, repn.U.b, M.U.inequalities)
        pyomo_util.add_linear_constraints(M.L, repn.L.A, repn.U, repn.L, repn.L.b, M.L.inequalities)

        # objective
        e = pyomo_util.dot(repn.U.c.U, repn.U, num=1) + pyomo_util.dot(repn.U.c.L, repn.L, num=1) + repn.U.d
        M.o = pe.Objective(expr=e)

        # dual variables for primal constraints
        M.Dual.dual_c = Var(range(len(M.L.c)))

        # duality gap
        e = pyomo_util._linear_expression(1, repn.c.L, repn.L)
        M.lower_gap = Constraint(expr=e[0] == M.z)

        return M

    def _debug(self):
        for j in M.U.xR:
            print("U",j,pe.value(M.U.xR[j]))
        for j in M.L.xR:
            print("L",j,pe.value(M.L.xR[j]))
        #for j in M.kkt.lam:
        #    print("lam",j,pe.value(M.kkt.lam[j]))
        #for j in M.kkt.nu:
        #    print("nu",j,pe.value(M.kkt.nu[j]))

