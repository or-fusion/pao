#
# A solver for linear bilevel programs using big-M 
# relaxations discussed by Fortuny-Amat and McCarl, 1981.
#
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.mpec import ComplementarityList, complements

import pao.common
from ..solver import Solver, LinearMultilevelSolverBase, LinearMultilevelResults
from ..repn import LinearMultilevelProblem
from ..convert_repn import convert_to_standard_form
from . import pyomo_util
from .reg import create_model_replacing_LL_with_kkt


@Solver.register(
        name='pao.mpr.FA',
        doc='PAO solver for Multilevel Problem Representations that define linear bilevel problems.  Solver uses big-M relaxations discussed by Fortuny-Amat and McCarl (1981).')

class LinearMultilevelSolver_FA(LinearMultilevelSolverBase):
    """
    PAO FA solver for linear MPRs: pao.mpr.FA

    This solver replaces lower-level problems using the KKT conditions and
    calls a MIP solver to solve the reformulated problem.
    """
    config = LinearMultilevelSolverBase.config()
    config.declare('mip_solver', ConfigValue(
        default='glpk',
        description="The MIP solver used by FA.  (default is glpk)"
        ))
    #config.declare('mip_options', ConfigValue(
    #    default=None,
    #    description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
    config.declare('bigm', ConfigValue(
        default=100000,
        domain=float,
        description="The big-M value used to enforce complementarity conditions.  (default is 1e5)"
        ))

    def __init__(self, **kwds):
        super().__init__(name='pao.mpr.FA')

    def check_model(self, model):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(model) is LinearMultilevelProblem), "Solver '%s' can only solve a linear multilevel problem" % self.name
        model.check()
        #
        # Confirm that this is a bilevel problem
        #
        for L in model.U.LL:
            assert (len(L.LL) == 0), "Can only solve bilevel problems"
        #
        # No binary or integer lower level variables
        #
        for L in model.U.LL:
            assert (L.x.nxZ == 0), "Cannot use solver %s with model with integer lower-level variables" % self.name
            assert (L.x.nxB == 0), "Cannot use solver %s with model with binary lower-level variables" % self.name

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

        self.standard_form, soln_manager = convert_to_standard_form(model, inequalities=False)

        M = self._create_pyomo_model(self.standard_form, self.config.bigm)
        #
        # Solve the Pyomo model the specified solver
        #
        results = LinearMultilevelResults(solution_manager=soln_manager)
        if isinstance(self.config.mip_solver, str):
            opt = pe.SolverFactory(self.config.mip_solver)
        else:
            opt = self.config.mip_solver

        #if self.config.mip_options is not None:
        #    opt.options.update(self.config.mip_options)
        pyomo_results = opt.solve(M, tee=self.config.tee, 
                                     load_solutions=self.config.load_solutions)
        pyomo.opt.check_optimal_termination(pyomo_results)

        self._initialize_results(results, pyomo_results, M)
        results.solver.rc = getattr(opt, '_rc', None)

        if self.config.load_solutions:
            # Load results from the Pyomo model to the LinearMultilevelProblem
            results.copy_solution(From=M, To=model)
        else:
            # Load results from the Pyomo model to the Results
            results.load_from(pyomo_results)

        #self._debug(M)
        #results.solver.log = getattr(opt, '_log', None)

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

    def _create_pyomo_model(self, repn, bigM):
        M = create_model_replacing_LL_with_kkt(repn)
        #
        # Transform the problem to a MIP
        #
        # TODO - directly create the bigM relaxation here.  Applying
        # Pyomo transformations in sequence creates a model object that is
        # difficult to interpret.
        #
        xfrm = pe.TransformationFactory('mpec.simple_disjunction')
        xfrm.apply_to(M)
        xfrm = pe.TransformationFactory('gdp.bigm')
        xfrm.apply_to(M, bigM=bigM)

        return M

    def _debug(self, M):    # pragma: no cover
        for j in M.U.xR:
            print("U",j,pe.value(M.U.xR[j]))
        for j in M.L.xR:
            print("L",j,pe.value(M.L.xR[j]))
        for j in M.kkt.lam:
            print("lam",j,pe.value(M.kkt.lam[j]))
        for j in M.kkt.nu:
            print("nu",j,pe.value(M.kkt.nu[j]))


pao.common.SolverAPI._generate_solve_docstring(LinearMultilevelSolver_FA)

