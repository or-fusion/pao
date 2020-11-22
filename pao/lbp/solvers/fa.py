#
# A solver for linear bilevel programs using big-M 
# relaxations discussed by Fortuny-Amat and McCarl, 1981.
#
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.mpec import ComplementarityList, complements
from ..solver import SolverFactory, LinearBilevelSolverBase, LinearBilevelResults
from ..repn import LinearBilevelProblem
from ..convert_repn import convert_LinearBilevelProblem_to_standard_form
from .. import pyomo_util


@SolverFactory.register(
        name='pao.lbp.FA',
        doc='A solver for linear bilevel programs using big-M relaxations discussed by Fortuny-Amat and McCarl, 1981.')
class LinearBilevelSolver_FA(LinearBilevelSolverBase):

    def __init__(self, **kwds):
        super().__init__(name='pao.lbp.FA')
        self.config.solver = 'glpk'
        self.config.bigm = 100000

    def check_model(self, lbp):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(lbp) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.name
        lbp.check()
        #
        # TODO: For now, we just deal with the case where there is a single lower-level.  Later, we
        # will generalize this.
        #
        assert (len(lbp.U.LL) == 1 and len(lbp.U.LL[0].LL) == 0), "Only one lower-level is handled right now"
        #
        # Confirm that this is a bilevel problem
        #
        for i in range(len(lbp.U.LL)):
            assert (len(lbp.U.LL[i].LL) == 0), "Can only solve bilevel problems"
        #
        # No binary or integer lower level variables
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

        M = self._create_pyomo_model(self.standard_form, self.config.bigm)
        #
        # Solve the Pyomo model the specified solver
        #
        results = LinearBilevelResults(solution_manager=soln_manager)
        with pe.SolverFactory(self.config.solver) as opt:
            if self.config.mipgap is not None:
                opt.options['mipgap'] = self.config.mipgap
            if options is not None:
                opt.options.update(options)
            pyomo_results = opt.solve(M, tee=self.config.tee, 
                                         timelimit=self.config.timelimit,
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

            #self._debug(M)
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

    def _create_pyomo_model(self, repn, bigM):
        U = repn.U
        L = U.LL[0]

        #
        # Create Pyomo model
        #
        M = pe.ConcreteModel()
        M.U = pe.Block()
        M.L = pe.Block()
        M.kkt = pe.Block()

        # upper- and lower-level variables
        pyomo_util.add_variables(M.U, U)
        pyomo_util.add_variables(M.L, L)
        # dual variables
        M.kkt.lam = pe.Var(range(len(L.b)))                                    # equality constraints
        M.kkt.nu = pe.Var(range(len(L.x)), within=pe.NonNegativeReals)        # variable bounds

        # objective
        e = pyomo_util.dot(U.c[U], U.x, num=1) + pyomo_util.dot(U.c[L], L.x, num=1) + U.d
        M.o = pe.Objective(expr=e)

        # upper-level constraints
        pyomo_util.add_linear_constraints(M.U, U.A, U, L, U.b, U.inequalities)
        # lower-level constraints
        pyomo_util.add_linear_constraints(M.L, L.A, U, L, L.b, L.inequalities)

        # stationarity
        M.kkt.stationarity = pe.ConstraintList() 
        # L_A_L' * lam
        L_A_L_T = L.A[L].transpose().todok()
        X = pyomo_util.dot( L_A_L_T, M.kkt.lam )
        for i in range(len(L.c[L])):
            #e = 0
            #for j in M.kkt.lam:
            #    e += L_A_L_xR_T[i,j] * M.kkt.lam[j]
            #M.kkt.stationarity.add( repn.L.c.L.xR[i] + e - M.kkt.nu[i] == 0 )
            M.kkt.stationarity.add( L.c[L][i] + X[i] - M.kkt.nu[i] == 0 )

        # complementarity slackness - variables
        M.kkt.slackness = ComplementarityList()
        for i in M.kkt.nu:
            M.kkt.slackness.add( complements( M.L.xR[i] >= 0, M.kkt.nu[i] >= 0 ) )

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

