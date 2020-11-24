#
# A solver for linear bilevel programs using regularization
# discussed by Scheel and Scholtes (2000) and Ralph and Wright (2004).
#
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.mpec import ComplementarityList, complements

import pao.common
from ..solver import SolverFactory, LinearBilevelSolverBase, LinearBilevelResults
from ..repn import LinearBilevelProblem
from ..convert_repn import convert_LinearBilevelProblem_to_standard_form
from .. import pyomo_util


def create_model_replacing_LL_with_kkt(repn):
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
    #e1 = pyomo_util.dot(U.c[U], U.x, num=1)
    #e2 = pyomo_util.dot(U.c[L], L.x, num=1)
    #e3 = U.d
    #print(type(e1), type(e2), type(e3))
    #print(e1, e2, e3)
    e = pyomo_util.dot(U.c[U], U.x, num=1) + U.d
    for i in range(N):
        L = LL[i]
        e += pyomo_util.dot(U.c[L], L.x, num=1)
    M.o = pe.Objective(expr=e)

    # upper-level constraints
    pyomo_util.add_linear_constraints(M.U, U.A, U, L, U.b, U.inequalities)
    # lower-level constraints
    for i in range(N):
        L = LL[i]
        pyomo_util.add_linear_constraints(M.L[i], L.A, U, L, L.b, L.inequalities)

    for i in range(N):
        L = LL[i]
        # stationarity
        M.kkt[i].stationarity = pe.ConstraintList() 
        # L_A_L' * lam
        L_A_L_T = L.A[L].transpose().todok()
        X = pyomo_util.dot( L_A_L_T, M.kkt[i].lam )
        for k in range(len(L.c[L])):
            M.kkt[i].stationarity.add( L.c[L][k] + X[k] - M.kkt[i].nu[k] == 0 )

        # complementarity slackness - variables
        for i in range(N):
            M.kkt[i].slackness = ComplementarityList()
            for j in M.kkt[i].nu:
                M.kkt[i].slackness.add( complements( M.L[i].xR[j] >= 0, M.kkt[i].nu[j] >= 0 ) )

    return M


@SolverFactory.register(
        name="pao.lbp.REG",
        doc="A solver for linear bilevel programs using regularization discussed by Scheel and Scholtes (2000) and Ralph and Wright (2004).")
class LinearBilevelSolver_REG(LinearBilevelSolverBase):

    def __init__(self, **kwds):
        super().__init__(name='pao.lbp.REG')
        self.config.solver = 'ipopt'
        self.config.rho = 1e-7

    def check_model(self, lbp):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(lbp) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.name
        lbp.check()
        #
        # Confirm that this is a bilevel problem
        #
        for i in range(len(lbp.U.LL)):
            assert (len(lbp.U.LL[i].LL) == 0), "Can only solve bilevel problems"
        #
        # No binary or integer upper-level variables
        #
        assert (lbp.U.x.nxZ == 0), "Cannot use solver %s with model with integer upper-level variables" % self.name
        assert (lbp.U.x.nxB == 0), "Cannot use solver %s with model with binary upper-level variables" % self.name
        #
        # No binary or integer lower-level variables
        #
        for L in lbp.U.LL:
            assert (L.x.nxZ == 0), "Cannot use solver %s with model with integer lower-level variables" % self.name
            assert (L.x.nxB == 0), "Cannot use solver %s with model with binary lower-level variables" % self.name

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

        self.standard_form, soln_manager = convert_LinearBilevelProblem_to_standard_form(lbp)

        M = self._create_pyomo_model(self.standard_form, self.config.rho)
        #
        # Solve the Pyomo model the specified solver
        #
        results = LinearBilevelResults(solution_manager=soln_manager)
        with pe.SolverFactory(self.config.solver) as opt:
            if options is not None:
                opt.options.update(options)
            pyomo_results = opt.solve(M, tee=self.config.tee, 
                                         timelimit=self.config.time_limit,
                                         load_solutions=self.config.load_solutions)
            pyomo.opt.check_optimal_termination(pyomo_results)

            self._initialize_results(results, pyomo_results, M)
            results.solver.rc = getattr(opt, '_rc', None)

            if self.config.load_solutions:
                # Load results from the Pyomo model to the LinearBilevelProblem
                results.copy_from_to(pyomo=M, lbp=lbp)
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

