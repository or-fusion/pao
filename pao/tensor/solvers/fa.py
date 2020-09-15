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
from ..solver import register_solver, LinearBilevelSolverBase
from ..repn import LinearBilevelProblem
from ..convert_repn import convert_LinearBilevelProblem_to_standard_form
from .. import pyomo_util


class LinearBilevelSolver_FA(LinearBilevelSolverBase):

    def __init__(self, **kwds):
        self.name = 'pao.lbp.FA'

    def check_model(self, lbp):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(lbp) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.solver_type
        lbp.check()
        #
        # TODO: For now, we just deal with the case where there is a single lower-level.  Later, we
        # will generalize this.
        #
        assert (len(lbp.L) == 1), "Only one lower-level is handled right now"
        #
        # No binary or integer lower level variables
        #
        for i in range(len(lbp.L)):
            assert (len(lbp.L[i].xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
            assert (len(lbp.L[i].xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type

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
        solver =    kwds.pop('solver', 'glpk')
        tee =       kwds.pop('tee', False)
        timelimit = kwds.pop('timelimit', None)
        mipgap =    kwds.pop('mipgap', None)
        bigM =      kwds.pop('bigM', 100000)
        #
        # Start clock
        #
        start_time = time.time()

        self.standard_form, self.multipliers = convert_LinearBilevelProblem_to_standard_form(lbp)

        M = self._create_pyomo_model(self.standard_form, bigM)
        #
        # Solve the Pyomo model the specified solver
        #
        with pe.SolverFactory(solver) as opt:
            if mipgap is not None:
                opt.options['mipgap'] = mipgap
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

    def _create_pyomo_model(self, repn, bigM):
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
        e = pyomo_util._linear_expression(1, repn.U.c.U, repn.U) + pyomo_util._linear_expression(1, repn.U.c.L, repn.L) + repn.U.d
        M.o = pe.Objective(expr=e[0])

        # upper-level constraints
        pyomo_util._linear_constraints(repn.U.inequalities, repn.U.A, repn.U, repn.L, repn.U.b, M.U)
        # lower-level constraints
        pyomo_util._linear_constraints(repn.L.inequalities, repn.L.A, repn.U, repn.L, repn.L.b, M.L)

        # stationarity
        L_A_L_xR_T = repn.L.A.L.xR.transpose().todok()
        M.kkt.stationarity = pe.ConstraintList() 
        for i in range(len(repn.L.c.L.xR)):
            # L_A_L_xR' * lam
            e = 0
            for j in M.kkt.lam:
                e += L_A_L_xR_T[i,j] * M.kkt.lam[j]
            M.kkt.stationarity.add( repn.L.c.L.xR[i] + e - M.kkt.nu[i] == 0 )

        # complementarity slackness - variables
        M.kkt.slackness = ComplementarityList()
        for i in M.kkt.nu:
            M.kkt.slackness.add( complements( M.L.xR[i] >= 0, M.kkt.nu[i] >= 0 ) )

        #
        # Transform the problem to a MIP
        #
        #M.pprint()
        xfrm = pe.TransformationFactory('mpec.simple_disjunction')
        xfrm.apply_to(M)
        xfrm = pe.TransformationFactory('gdp.bigm')
        xfrm.apply_to(M, bigM=bigM)
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


register_solver('pao.lbp.FA', LinearBilevelSolver_FA)
