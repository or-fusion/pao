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
from ..solver import LinearBilevelSolver, LinearBilevelSolverBase
from ..repn import LinearBilevelProblem
from ..convert_repn import convert_LinearBilevelProblem_to_standard_form
from .. import pyomo_util


class LinearBilevelSolver_REG(LinearBilevelSolverBase):

    def __init__(self, **kwds):
        self.name = 'pao.lbp.REG'

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
        assert (len(lbp.L) == 1), "Only one lower-level is handled right now"
        #
        # No binary or integer upper-level variables
        #
        assert (len(lbp.U.xZ) == 0), "Cannot use solver %s with model with integer upper-level variables" % self.name
        assert (len(lbp.U.xB) == 0), "Cannot use solver %s with model with binary upper-level variables" % self.name
        #
        # No binary or integer lower-level variables
        #
        for i in range(len(lbp.L)):
            assert (len(lbp.L[i].xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.name
            assert (len(lbp.L[i].xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.name

    def solve(self, *args, **kwds):
        #
        # Error checks
        #
        assert (len(args) == 1), "Can only solve a single LinearBilevelProblem"
        lbp = args[0]
        assert (lbp.__class__ == LinearBilevelProblem), "Unexpected argument of type %s" % str(type(lbp))
        self.check_model(lbp)
        #
        # Process keyword options
        #
        solver =    kwds.pop('solver', 'ipopt')
        tee =       kwds.pop('tee', False)
        timelimit = kwds.pop('timelimit', None)
        rho =      kwds.pop('rho', 1e-7)
        #
        # Start clock
        #
        start_time = time.time()

        self.standard_form, self.multipliers = convert_LinearBilevelProblem_to_standard_form(lbp)

        M = self._create_pyomo_model(self.standard_form, rho)
        #
        # Solve the Pyomo model the specified solver
        #
        with pe.SolverFactory(solver) as opt:
            results = opt.solve(M, tee=tee, timelimit=timelimit)

            pyomo.opt.check_optimal_termination(results)
            self._update_solution(lbp, M)
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            #self.results_obj = self._setup_results_obj()
            #
            # Return the sub-solver return condition value and log
            #
            return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                       log=getattr(opt, '_log', None))

    def _setup_results_obj(self):
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
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        return results

    def _create_pyomo_model(self, repn, rho):
        #repn.print()
        #print("*"*80)
        #
        # Create Pyomo model
        #
        M = pe.ConcreteModel()
        M.U = pe.Block()
        M.L = pe.Block()
        M.kkt = pe.Block()

        # upper- and lower-level variables
        pyomo_util._create_variables(repn.U, M.U)
        pyomo_util._create_variables(repn.L, M.L)
        # dual variables
        M.kkt.lam = pe.Var(range(len(repn.L.b)))                                    # equality constraints
        M.kkt.nu = pe.Var(range(len(repn.L.xR)), within=pe.NonNegativeReals)        # variable bounds

        # objective
        e = pyomo_util.dot(repn.U.c.U, repn.U, num=1) + pyomo_util.dot(repn.U.c.L, repn.L, num=1) + repn.U.d
        M.o = pe.Objective(expr=e)

        # upper-level constraints
        pyomo_util.add_linear_constraints(M.U, repn.U.A, repn.U, repn.L, repn.U.b, repn.U.inequalities)
        # lower-level constraints
        pyomo_util.add_linear_constraints(M.L, repn.L.A, repn.U, repn.L, repn.L.b, repn.L.inequalities)

        # stationarity
        M.kkt.stationarity = pe.ConstraintList() 
        # L_A_L_xR' * lam
        L_A_L_xR_T = repn.L.A.L.xR.transpose().todok()
        X = pyomo_util.dot( L_A_L_xR_T, M.kkt.lam )
        for i in range(len(repn.L.c.L.xR)):
            #e = 0
            #for j in M.kkt.lam:
            #    e += L_A_L_xR_T[i,j] * M.kkt.lam[j]
            #M.kkt.stationarity.add( repn.L.c.L.xR[i] + e - M.kkt.nu[i] == 0 )
            M.kkt.stationarity.add( repn.L.c.L.xR[i] + X[i] - M.kkt.nu[i] == 0 )

        # complementarity slackness - variables
        M.kkt.slackness = ComplementarityList()
        for i in M.kkt.nu:
            M.kkt.slackness.add( complements( M.L.xR[i] >= 0, M.kkt.nu[i] >= 0 ) )

        #
        # Transform the problem to a MIP
        #
        #M.pprint()
        xfrm = pe.TransformationFactory('mpec.simple_nonlinear')
        xfrm.apply_to(M, mpec_bound=rho)
        #print("="*80)
        #M.pprint()
        #M.display()
        #print("="*80)

        return M

    def _update_solution(self, repn, M):
        if False:
            for j in M.U.xR:
                print("U",j,pe.value(M.U.xR[j]))
            for j in M.L.xR:
                print("L",j,pe.value(M.L.xR[j]))
            for j in M.kkt.lam:
                print("lam",j,pe.value(M.kkt.lam[j]))
            for j in M.kkt.nu:
                print("nu",j,pe.value(M.kkt.nu[j]))

        for j in repn.U.xR:
            repn.U.xR.values[j] = sum(pe.value(M.U.xR[v]) * c for v,c in self.multipliers[0][j])
        for j in repn.U.xZ:
            repn.U.xZ.values[j] = pe.value(M.U.xZ[j])
        for j in repn.U.xB:
            repn.U.xB.values[j] = pe.value(M.U.xB[j])
        #
        # TODO - generalize to multiple sub-problems
        #
        for i in range(len(repn.L)):
            for j in repn.L[i].xR:
                repn.L[i].xR.values[j] = sum(pe.value(M.L.xR[v]) * c for v,c in self.multipliers[1][i][j])
            for j in repn.L[i].xZ:
                repn.L[i].xZ.values[j] = pe.value(M.L.xZ[j])
            for j in repn.L[i].xB:
                repn.L[i].xB.values[j] = pe.value(M.L.xB[j])


LinearBilevelSolver.register('pao.lbp.REG', LinearBilevelSolver_REG)
