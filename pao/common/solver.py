#
# Classes used to define a solver API
#
import six
import abc
import enum

"""
An enumeration used to define the termination condition of solvers
"""
class TerminationCondition(enum.Enum):

    """unknown serves as both a default value, and it is used when no other enum member makes sense"""
    unknown = 0

    """The solver exited due to a time limit"""
    maxTimeLimit = 1

    """The solver exited due to an iteration limit """
    maxIterations = 2

    """The solver exited due to an objective limit"""
    objectiveLimit = 3

    """The solver exited due to a minimum step length"""
    minStepLength = 4

    """The solver exited with an optimal solution"""
    optimal = 5

    """The solver exited because the problem is unbounded"""
    unbounded = 8

    """The solver exited because the problem is infeasible"""
    infeasible = 9

    """The solver exited because the problem is either infeasible or unbounded"""
    infeasibleOrUnbounded = 10

    """The solver exited due to an error"""
    error = 11

    """The solver exited because it was interrupted"""
    interrupted = 12

    """The solver exited due to licensing problems"""
    licensingProblems = 13


"""
An API for optimization solvers
"""
class Solver(six.with_metaclass(abc.ABCMeta, object)):

    def __init__(self, **kwds):
        self.options = Bunch(
            time_limit=None,
            keepfiles=False,
            tee=False,
            load_solution=True
            )

    """
    Returns True if the solver can be executed.
    """
    def available(self):
        True

    """
    Returns True if the solver meets license requirements for execution.
    """
    def license_status(self):
        True

    """
    Returns a tuple describing the solver version.
    """
    def version(self):
        return tuple()

    """
    Execute the solver and load the solution into the model.

    TODO: description options vs. config_options
    """
    @abc.abstractmethod
    def solve(self, model, options=None, **config_options):
        pass
    

    """
    Returns True if the solver is persistent and supports resolve logic.
    """
    def is_persistent():
        return False

    """
    Returns an error.
    """
    def __bool__(self):
        raise RuntimeError("Casting a solver to bool() is not allowed.  Use available() to check if the solver can be executed.")



"""
A results object is returned from the Solver.solve() method.  This
object reports information from a solver's execution, including the
termination condition, objective value for the best feasible point
and the best bound computed on the objective value.  Additionally,
a results object may contain solutions found during optimization.

Here is an example workflow:

>>> opt = SolverFactory('my_solver')
>>> results = opt.solve(my_model, load_solution=False)
>>> if results.solver.termination_condition == TerminationCondition.optimal:
>>>     print('optimal solution found: ', results.solver.best_feasible_objective)
>>>     results.load_solution( my_model )
>>>     print('the optimal value of x is ', my_model.x.value)
>>> elif results.found_feasible_solution():
>>>     print('sub-optimal but feasible solution found: ', results.solver.best_feasible_objective)
>>>     results.load_vars(vars_to_load=[my_model.x])
>>>     print('The value of x in the feasible solution is ', my_model.x.value)
>>> elif results.solver.termination_condition in {TerminationCondition.maxIterations,
...                                               TerminationCondition.maxTimeLimit}:
>>>     print('No feasible solution was found. The best lower bound found was ',
...           results.solver.best_objective_bound)
>>> else:
>>>     print('The following termination condition was encountered: ',
...           results.solver.termination_condition)
"""
class ResultsBase(six.with_metaclass(abc.ABCMeta, object)):

    def __init__(self):
        self._solution_loader = None
        self.solver = Bunch(
                termination_condition=TerminationCondition.unknown,
                best_feasible_objective=None,
                )
        self.problem = Bunch()

    """
    Load the the solution from a results object into the 
    specified model.
    """
    def load_solution(self, model):
        self._solution_loader.load_solution(model)

    """
    Returns
    -------
    found_feasible_solution: bool
        True if at least one feasible solution was found. False otherwise.
    """
    @abc.abstractmethod
    def found_feasible_solution(self):
        return self._found_feasible_solution


#
# Keep this?
#
class Results(ResultsBase):

    def __init__(self, found_feasible_solution):
        super(Results, self).__init__()
        self._found_feasible_solution = found_feasible_solution

    def found_feasible_solution(self):
        return self._found_feasible_solution


#
# A class for managing global registry of solvers
#
class SolverFactoryClass(object):

    _registry = {}

    """
    Register a solver class with the specified name.
    """
    def register(self, name, cls):
        SolverFactoryClass._registry[name] = cls

    """
    Construct a Solver class instance that is registered with the
    specified name.
    """
    def __call__(self, name):
        assert (name in SolverFactoryClass._registry), "Unknown solver '%s' specified" % name
        return SolverFactoryClass._registry[name]()


SolverFactory = SolverFactoryClass()
