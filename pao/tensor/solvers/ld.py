import numpy as np
import pyomo.environ as pe
#import pao.bilevel
#import pao.bilevel.plugins
from ..solver import register_solver, LinearBilevelSolverBase
from ..repn import LinearBilevelProblem


class LinearBilevelSolver_ld(LinearBilevelSolverBase):

    def __init__(self, **kwds):
        self.name = 'pao.lbp.ld'

    def check_model(self, M):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(M) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.solver_type
        M.check()
        #
        # No binary or integer lower level variables
        #
        assert (len(M.L.xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
        assert (len(M.L.xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type
        #
        # Upper and lower objectives are the opposite of each other
        #
        # TODO

    def solve(self, *args, **kwds):
        #
        # Error checks
        #
        assert (len(args) == 1), "Can only solve a single LinearBilevelProblem"
        repn = args[0]
        assert (repn.__class__ == LinearBilevelProblem), "Unexpected argument of type %s" % str(type(lbp))
        self.check_model(repn)
        #
        # Process keyword options
        #
        solver =    kwds.pop('solver', 'glpk')
        tee =       kwds.pop('tee', False)
        timelimit = kwds.pop('timelimit', None)
        mipgap =    kwds.pop('mipgap', None)
        #
        # Start clock
        #
        start_time = time.time()

        M = self._create_pyomo_model(repn)
        #
        # Solve the Pyomo model the specified solver
        #
        with pyomo.opt.SolverFactory(solver) as opt:
            if mipgap is not None:
                opt.options['mipgap'] = mipgap
            results = opt.solve(M, tee=tee, timelimit=timelimit)

            pyomo_util.check_termination(results.solver.termination_condition)

            # check that the solutions list is not empty
            if M.solutions.solutions:
                M.solutions.select(0, ignore_fixed_vars=True)
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            self.results_obj = self._setup_results_obj()

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

    def _create_pyomo_model(self, repn):
        #
        # Create Pyomo model
        #
        M = pe.ConcreteModel()
        M.z = pe.Var()
        M.U = pe.Block()
        M.L = pe.Block()
        M.Dual = pe.Block()

        # upper- and lower-level variables
        pyomo_util._create_variables(repn.U, M.U)
        pyomo_util._create_variables(repn.L, M.L)

        # upper- and lower-level constraints
        pyomo_util._linear_constraints(repn.U.inequalities, repn.U.A, repn.U, repn.L, repn.U.b, M.U)
        pyomo_util._linear_constraints(repn.L.inequalities, repn.L.A, repn.U, repn.L, repn.L.b, M.L)

        # objective
        e = pyomo_util._linear_expression(1, repn.c.U, repn.U) + M.z
        if repn.U.minimize:
            M.o = pe.Objective(expr=e[0])
        else:
            M.o = pe.Objective(expr=e[0], sense=pe.maximize)

        # dual variables for primal constraints
        M.Dual.dual_c = Var(range(len(M.L.c)))
        # dual variables for primal lower-bounds
        M.Dual.dual_vlb = Var([i for i,v in M.L.xR if v.bounds[0] is not None])
        # dual variables for primal upper-bounds
        M.Dual.dual_vub = Var([i for i,v in M.L.xR if v.bounds[1] is not None])

        # duality gap
        e = pyomo_util._linear_expression(1, repn.c.L, repn.L)
        M.lower_gap = Constraint(expr=e[0] == M.z)

        return M


register_solver('pao.lbp.ld', LinearBilevelSolver_ld)
