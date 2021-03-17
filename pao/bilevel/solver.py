import time
from .convert import convert_pyomo2MultilevelProblem
import pao.common
import pao.lbp


SolverFactory = pao.common.SolverFactory  


class PyomoSubmodelSolverBase(pao.common.Solver):
    """
    API for solvers that optimize a Pyomo model using SubModel components.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def Xcheck_model(self, problem):         # pragma: no cover
        # Confirm that the compact problem representation is well-formed.
        pass


class PyomoSubmodelSolverBase_LBP(PyomoSubmodelSolverBase):
    """
    Define the API for solvers that optimize a Pyomo model using SubModel components
    """

    def __init__(self, name, lmp_solver, inequalities):
        super().__init__(name)
        self.lmp_solver = lmp_solver
        self.inequalities = inequalities

    def Xinequalities(self):
        #
        # Return True if the conversion to LinearMultilevelProblem should
        # use inequalities (True) or equalities (False)
        #
        return False

    def solve(self, model, **options):
        #
        # Process keyword options
        #
        options = self._update_config(options, validate_options=False)
        solver_options = {k:self.config[k] for k in self.config}
        solver_options['load_solutions'] = True
        linearize_bilevel = options.get('linearize_bilinear_terms',False)
        #
        # Start the clock
        #
        start_time = time.time()
        #
        # Convert the Pyomo model to a LBP
        #
        try:
            mp, soln_manager = convert_pyomo2MultilevelProblem(model)
        except RuntimeError as err:
            print("Cannot convert Pyomo model to a multilevel problem") 
            raise
        if linearize_bilevel:
            lmp = pao.lbp.linearize_bilinear_terms(mp)
        else:
            lmp = mp
        #
        results = PyomoSubmodelResults(solution_manager=soln_manager)
        with SolverFactory(self.lmp_solver) as opt:
            lmp_results = opt.solve(lmp, **solver_options)

            self._initialize_results(results, lmp_results, model, lmp, options)
            results.solver.rc = getattr(opt, '_rc', None)
            results.copy_from_to(lbp=lmp, pyomo=model)
            
        results.solver.wallclock_time = time.time() - start_time
        return results

    def _initialize_results(self, results, lmp_results, instance, lmp, options):
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.name
        solv.lmp_solver = self.lmp_solver
        solv.config = self.config
        solv.termination_condition = lmp_results.solver.termination_condition
        solv.solver_time = lmp_results.solver.time
        solv.best_feasible_objective = lmp_results.solver.best_feasible_objective
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
        prob.lower_bound = lmp_results.problem.lower_bound
        prob.upper_bound = lmp_results.problem.upper_bound
        prob.sense = lmp_results.problem.sense

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

