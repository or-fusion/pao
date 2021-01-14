import time
from .convert import convert_pyomo2LinearBilevelProblem
import pao.common
import pao.lbp


SolverFactory = pao.common.SolverFactory  


class PyomoSubmodelSolverBase(pao.common.Solver):
    """
    Define the API for solvers that optimize a Pyomo model using SubModel components
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def check_model(self, lbp):         # pragma: no cover
        #
        # Confirm that the problem is well-formed
        #
        pass

    def solve(self, *args, **kwds):     # pragma: no cover
        #
        # Solve the Pyomo model
        #
        pass


class PyomoSubmodelSolverBase_LBP(PyomoSubmodelSolverBase):
    """
    Define the API for solvers that optimize a Pyomo model using SubModel components
    """

    def __init__(self, name, lbp_solver, inequalities):
        super().__init__(name)
        self.lbp_solver = lbp_solver
        self.inequalities = inequalities

    def inequalities(self):
        #
        # Return True if the conversion to LinearBilevelProblem should
        # use inequalities (True) or equalities (False)
        #
        return False

    def solve(self, instance, options=None, **config_options):
        #
        # Process keyword options
        #
        #for key, value in config_options.items():
        #    setattr(self.config, key, value)
        config_options = self._update_config(config_options, validate_options=False)
        #
        # Start the clock
        #
        start_time = time.time()
        #
        # Convert the Pyomo model to a LBP
        #
        try:
            lbp, soln_manager = convert_pyomo2LinearBilevelProblem(instance)
        except RuntimeError as err:
            print("Cannot convert Pyomo model to a LinearBilevelProblem")
            raise
        #
        results = PyomoSubmodelResults(solution_manager=soln_manager)
        with SolverFactory(self.lbp_solver) as opt:
            lbp_results = opt.solve(lbp, options=options, 
                                        load_solutions=True,
                                        **config_options
                                        )

            self._initialize_results(results, lbp_results, instance, lbp, options)
            results.solver.rc = getattr(opt, '_rc', None)
            results.copy_from_to(lbp=lbp, pyomo=instance)
            
        results.solver.wallclock_time = time.time() - start_time
        return results

    def _initialize_results(self, results, lbp_results, instance, lbp, options):
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.name
        solv.lbp_solver = self.lbp_solver
        solv.config = self.config
        solv.solver_options = options
        solv.termination_condition = lbp_results.solver.termination_condition
        solv.solver_time = lbp_results.solver.time
        solv.best_feasible_objective = lbp_results.solver.best_feasible_objective
        #
        # PROBLEM
        #
        prob = results.problem
        prob.name = instance.name
        prob.number_of_constraints = instance.statistics.number_of_constraints
        prob.number_of_variables = instance.statistics.number_of_variables
        prob.number_of_binary_variables = instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables = instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables = instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = instance.statistics.number_of_objectives
        prob.lower_bound = lbp_results.problem.lower_bound
        prob.upper_bound = lbp_results.problem.upper_bound
        prob.sense = lbp_results.problem.sense

        return results


class PyomoSubmodelResults(pao.common.Results):

    def __init__(self, solution_manager=None):
        super(pao.common.Results, self).__init__()
        self._solution_manager=solution_manager

    def copy_from_to(self, **kwds):
        self._solution_manager.copy_from_to(**kwds)

    def load_from(self, data):          # pragma: no cover
        assert (False), "load_from() is not implemented"
        self._solution_manager.load_from(data)

