#
# A solver interface to MIBS
#
# A branch-and-cut algorithm for mixed integer bilevel linear optimization problems and its implementation
# S Tahernejad, TK Ralphs, ST DeNegre
# Mathematical Programming Computation 12 (4), 529-568
#
# TODO - Citations
#
import sys
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.common.config import ConfigBlock, ConfigValue
#from pyomo.mpec import ComplementarityList, complements

import pao.common
from ..solver import Solver, LinearMultilevelSolverBase, LinearMultilevelResults
from ..repn import LinearMultilevelProblem
from ..convert_repn import convert_to_standard_form
from . import pyomo_util
#from .reg import create_model_replacing_LL_with_kkt


@Solver.register(
        name='pao.mpr.MIBS',
        doc='PAO solver for Multilevel Problem Representations using the COIN-OR MIBS solver by Tahernejad, Ralphs, and DeNegre (2020).')

class LinearMultilevelSolver_MIBS(LinearMultilevelSolverBase):
    """
    PAO MIBS solver for linear MPRs: pao.mpr.MIBS
    """
    config = LinearMultilevelSolverBase.config()
    config.declare('executable', ConfigValue(
        default='mibs',
        description="The executable used for MIBS.  (default is mibs)"
        ))

    def __init__(self, **kwds):
        super().__init__(name='pao.mpr.MIBS')

    def check_model(self, model):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(model) is LinearMultilevelProblem), "Solver '%s' can only solve a linear multilevel problem" % self.name
        model.check()
        #
        # Confirm that this is a bilevel problem with just one lower-level
        #
        for L in model.U.LL:
            assert (len(L.LL) == 0), "Can only solve bilevel problems"
        assert (len(model.U.LL) == 1), "Can only solve a bilevel problem with a single lower-level"

    def solve(self, model, **options):
        #
        # Error checks
        #
        self.check_model(model)
        #
        # Process keyword options
        #
        self._update_config(options)
        #
        # Start clock
        #
        start_time = time.time()

        self.standard_form, soln_manager = convert_to_standard_form(model, inequalities=True)

        #
        # Write the MPS file and MIBS auxilliary file
        #
        # TODO - Make these temporary files
        #
        M = self.create_mibs_model(model, "mibs.mps", "mibs.aux")

        sys.exit(0)
        #
        # Launch solver
        #

        #pyomo_results = opt.solve(M, tee=self.config.tee, 
        #                             load_solutions=self.config.load_solutions)
        #pyomo.opt.check_optimal_termination(pyomo_results)
        #
        #self._initialize_results(results, pyomo_results, M)
        #results.solver.rc = getattr(opt, '_rc', None)

        if self.config.load_solutions:
            # Load results from the Pyomo model to the LinearMultilevelProblem
            results.copy_solution(From=M, To=model)
        else:
            # Load results from the Pyomo model to the Results
            results.load_from(pyomo_results)

        results.solver.wallclock_time = time.time() - start_time
        return results

    def _initialize_results(self, results, pyomo_results, M):
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.config.mip_solver
        solv.termination_condition = pyomo_util.pyomo2pao_termination_condition(pyomo_results.solver.termination_condition)
        if hasattr(pyomo_results.solver, 'time'):
            solv.solver_time = pyomo_results.solver.time
        if self.config.load_solutions:
            solv.best_feasible_objective = pe.value(M.o)
        #
        # PROBLEM - Maybe this should be the summary of the BLP itself?
        #
        prob = results.problem
        prob.name = M.name
        #prob.number_of_objectives = pyomo_results.problem.Number_of_objectives
        #prob.number_of_constraints = pyomo_results.problem.Number_of_constraints
        #prob.number_of_variables = pyomo_results.problem.Number_of_variables
        #prob.number_of_nonzeros = pyomo_results.problem.Number_of_nonzeros
        #prob.lower_bound = pyomo_results.problem.Lower_bound
        #prob.upper_bound = pyomo_results.problem.Upper_bound
        #prob.sense = 'minimize'
        return results

    def _debug(self, M):    # pragma: no cover
        for j in M.U.xR:
            print("U",j,pe.value(M.U.xR[j]))
        for j in M.L.xR:
            print("L",j,pe.value(M.L.xR[j]))
        for j in M.kkt.lam:
            print("lam",j,pe.value(M.kkt.lam[j]))
        for j in M.kkt.nu:
            print("nu",j,pe.value(M.kkt.nu[j]))





    def create_mibs_model(self, repn, mps_filename, aux_filename):
        """
        TODO - Document this transformation
        """
        U = repn.U
        L = repn.U.LL[0]

        #---------------------------------------------------
        # Create Pyomo model
        #---------------------------------------------------

        M = pe.ConcreteModel()
        M.U = pe.Block()
        M.L = pe.Block()

        # upper-level variables
        pyomo_util.add_variables(M.U, U)
        # lower-level variables
        pyomo_util.add_variables(M.L, L)

        # objective
        e = pyomo_util.dot(U.c[U], U.x, num=1) + U.d
        e += pyomo_util.dot(U.c[L], L.x, num=1)
        M.o = pe.Objective(expr=e)

        # upper-level constraints
        pyomo_util.add_linear_constraints(M.U, U.A, U, L, U.b, U.inequalities)
        # lower-level constraints
        pyomo_util.add_linear_constraints(M.L, L.A, U, L, L.b, L.inequalities)

        #---------------------------------------------------
        # Write files
        #---------------------------------------------------
        # TODO - get variable mapping information
        #
        M.write(mps_filename)

        with open(aux_filename, "w") as OUTPUT:
            # Num lower-level variables
            OUTPUT.write("N {}\n".format(len(L.x)))
            # Num lower-level constraints
            OUTPUT.write("M {}\n".format(L.b.size))
            # Indices of lower-level variables
            nx_upper = len(U.x)
            for i in range(len(L.x)):
                OUTPUT.write("LC {}\n".format(i+nx_upper))
            # Indices of lower-level constraints
            nc_upper = U.b.size
            for i in range(L.b.size):
                OUTPUT.write("LR {}\n".format(i+nc_upper))
            # Coefficients for lower-level objective
            for i in range(len(L.x)):
                OUTPUT.write("LO {}\n".format(L.c[L][i]))
            # Lower-level objective sense
            if L.minimize:
                OUTPUT.write("OS 1\n")
            else:
                OUTPUT.write("OS -1\n")
        
        return M

pao.common.SolverAPI._generate_solve_docstring(LinearMultilevelSolver_MIBS)
