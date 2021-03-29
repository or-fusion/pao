#
# A solver for linear bilevel programs using regularization
# discussed by Scheel and Scholtes (2000) and Ralph and Wright (2004).
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


def create_model_replacing_LL_with_kkt(repn):
    """
    TODO - Document this transformation
    """
    U = repn.U
    LL = repn.U.LL
    N = len(LL)

    #
    # Create Pyomo model
    #
    M = pe.ConcreteModel()
    M.U = pe.Block()
    M.L = pe.Block(range(N))
    M.kkt = pe.Block(range(N))

    # upper- and lower-level variables
    pyomo_util.add_variables(M.U, U)
    for i in range(N):
        L = LL[i]
        # lower-level variables
        pyomo_util.add_variables(M.L[i], L)
        # dual variables
        M.kkt[i].lam = pe.Var(range(len(L.b)))                                    # equality constraints
        M.kkt[i].nu = pe.Var(range(len(L.x)), within=pe.NonNegativeReals)         # variable bounds

    # objective
    e = pyomo_util.dot(U.c[U], U.x, num=1) + U.d
    for i in range(N):
        L = LL[i]
        e += pyomo_util.dot(U.c[L], L.x, num=1)
    M.o = pe.Objective(expr=e)

    # upper-level constraints
    pyomo_util.add_linear_constraints(M.U, U.A, U, L, U.b, U.inequalities)
    for i in range(N):
        # lower-level constraints
        L = LL[i]
        pyomo_util.add_linear_constraints(M.L[i], L.A, U, L, L.b, L.inequalities)

    for i in range(N):
        L = LL[i]
        # stationarity
        M.kkt[i].stationarity = pe.ConstraintList() 
        # L_A_L' * lam
        L_A_L_T = L.A[L].transpose().todok()
        X = pyomo_util.dot( L_A_L_T, M.kkt[i].lam )
        if L.c[L] is not None:
            for k in range(len(L.c[L])):
                M.kkt[i].stationarity.add( L.c[L][k] + X[k] - M.kkt[i].nu[k] == 0 )

    for i in range(N):
        # complementarity slackness - variables
        M.kkt[i].slackness = ComplementarityList()
        for j in M.kkt[i].nu:
            M.kkt[i].slackness.add( complements( M.L[i].xR[j] >= 0, M.kkt[i].nu[j] >= 0 ) )

    return M


@Solver.register(
        name="pao.mpr.REG",
        doc="PAO solver for Multilevel Problem Representations that define linear bilevel problems.  Solver uses regularization discussed by Scheel and Scholtes (2000) and Ralph and Wright (2004).")
class LinearMultilevelSolver_REG(LinearMultilevelSolverBase):
    """
    PAO REG solver for linear MPRs: pao.mpr.REG

    This solver replaces lower-level problems using the KKT conditions and
    calls a NLP solver to solve the reformulated problem.
    """

    config = LinearMultilevelSolverBase.config()
    config.declare('nlp_solver', ConfigValue(
        default='ipopt',
        description="The NLP solver used by REG.  (default is ipopt)"
        ))
    #config.declare('nlp_options', ConfigValue(
    #    default=None,
    #    description="A dictionary that defines the solver options for the NLP solver.  (default is None)"))
    config.declare('rho', ConfigValue(
        default=1e-7,
        domain=float,
        description="The tolerance for constraints that enforce complementarity conditions.  (default is 1e-7)"
        ))

    def __init__(self, **kwds):
        super().__init__(name='pao.mpr.REG')

    def check_model(self, mpr):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(mpr) is LinearMultilevelProblem), "Solver '%s' can only solve a LinearMultilevelProblem" % self.name
        mpr.check()
        #
        # Confirm that this is a bilevel problem
        #
        for i in range(len(mpr.U.LL)):
            assert (len(mpr.U.LL[i].LL) == 0), "Can only solve bilevel problems"
        #
        # No binary or integer upper-level variables
        #
        assert (mpr.U.x.nxZ == 0), "Cannot use solver %s with model with integer upper-level variables" % self.name
        assert (mpr.U.x.nxB == 0), "Cannot use solver %s with model with binary upper-level variables" % self.name
        #
        # No binary or integer lower-level variables
        #
        for L in mpr.U.LL:
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

        M = self._create_pyomo_model(self.standard_form, self.config.rho)
        #
        # Solve the Pyomo model the specified solver
        #
        results = LinearMultilevelResults(solution_manager=soln_manager)
        if isinstance(self.config.nlp_solver, str):
            opt = pe.SolverFactory(self.config.nlp_solver)
        else:
            opt = self.config.nlp_solver

        #if self.config.nlp_options is not None:
        #    opt.options.update(self.config.nlp_options)
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

        #self._debug()
        #results.solver.log = getattr(opt, '_log', None)

        results.solver.wallclock_time = time.time() - start_time
        return results

    def _initialize_results(self, results, pyomo_results, M):
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.config.nlp_solver
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
        prob.number_of_objectives = pyomo_results.problem.Number_of_objectives
        prob.number_of_constraints = pyomo_results.problem.Number_of_constraints
        prob.number_of_variables = pyomo_results.problem.Number_of_variables
        #prob.number_of_nonzeros = pyomo_results.problem.Number_of_nonzeros
        prob.lower_bound = pyomo_results.problem.Lower_bound
        prob.upper_bound = pyomo_results.problem.Upper_bound
        prob.sense = 'minimize'
        return results

    def _create_pyomo_model(self, repn, rho):
        M = create_model_replacing_LL_with_kkt(repn)
        #
        # Transform the problem to a MIP
        #
        xfrm = pe.TransformationFactory('mpec.simple_nonlinear')
        xfrm.apply_to(M, mpec_bound=rho)

        return M

    def _debug(self):           # pragma: no cover
        for j in M.U.xR:
            print("U",j,pe.value(M.U.xR[j]))
        for j in M.L.xR:
            print("L",j,pe.value(M.L.xR[j]))
        for j in M.kkt.lam:
            print("lam",j,pe.value(M.kkt.lam[j]))
        for j in M.kkt.nu:
            print("nu",j,pe.value(M.kkt.nu[j]))


pao.common.SolverAPI._generate_solve_docstring(LinearMultilevelSolver_REG)
