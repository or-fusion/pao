#
# Classes used to define a solver API
#
import six
import abc
import enum
from pyutilib.misc import Options


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
        self.config = Options(
            time_limit=None,
            keepfiles=False,
            tee=False,
            load_solutions=True
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

    def _update_config(self, config_options, validate_options=True):
        keys = set(config_options.keys())
        for k,v in config_options.items():
            if k in self.config:
                self.config[k] = v
                keys.remove(k)    
        if validate_options:
            assert (len(keys) == 0), "Unexpected options to solve() have been specified: %s" % " ".join(sorted(k for k in keys))
        return {key:config_options[key] for key in keys}

    """
    Support "with" statements.
    """
    def __enter__(self):
        return self

    # TODO
    def __exit__(self, t, v, traceback):
        pass


"""
A results object is returned from the Solver.solve() method.  This
object reports information from a solver's execution, including the
termination condition, objective value for the best feasible point
and the best bound computed on the objective value.  Additionally,
a results object may contain solutions found during optimization.

Here is an example workflow:

>>> opt = SolverFactory('my_solver')
>>> results = opt.solve(my_model, load_solutions=False)
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
        self._solution_manager = None
        self.solver = Options(
                termination_condition=TerminationCondition.unknown,
                best_feasible_objective=None,
                )
        self.problem = Options()

    """
    Returns
    -------
    found_feasible_solution: bool
        True if at least one feasible solution was found. False otherwise.
    """
    @abc.abstractmethod
    def found_feasible_solution(self):
        pass

    """
    Store the solution in the results object into the 
    model.
    """
    @abc.abstractmethod
    def store_to(self, model, i=0):
        pass

    """
    Load solution from the model.
    """
    @abc.abstractmethod
    def load_from(self, model):
        pass

    """
    Generate a string summary of the results object
    """
    @abc.abstractmethod
    def __str__(self):
        pass


class Results(ResultsBase):

    def __init__(self, found_feasible_solution=None):
        super(Results, self).__init__()
        self._found_feasible_solution = found_feasible_solution

    def found_feasible_solution(self):
        return self._found_feasible_solution

    def store_to(self, model, i=0):
        self._solution_manager.store_to(model, i=0)

    def __str__(self):
        problem = self.problem.__str__(indent='  ')
        solver = self.solver.__str__(indent='  ')
        return "Problem:\n-" + problem[1:] + \
            "\nSolver:\n-" + solver[1:]


#
# A class for managing global registry of solvers
#
class SolverFactoryClass(object):

    _registry = {}
    _doc = {}

    """
    Register a solver class with the specified name.
    """
    def register(self, cls=None, *, name=None, doc=None):
        def decorator(cls):
            assert (name is not None), "Must register a solver with a name"
            SolverFactoryClass._registry[name] = cls
            SolverFactoryClass._doc[name] = doc
        if cls is None:
            return decorator
        return decorator(cls)

    """
    Iterator showing all the solver names and descriptions
    """
    def __iter__(self):
        for name in sorted(SolverFactoryClass._doc.keys()):
            yield name

    """
    Method used to get the description of a solver
    """
    def doc(self, name):
        return SolverFactoryClass._doc[name]

    """
    Construct a Solver class instance that is registered with the
    specified name.
    """
    def __call__(self, name):
        assert (name in SolverFactoryClass._registry), "Unknown solver '%s' specified" % name
        return SolverFactoryClass._registry[name]()


SolverFactory = SolverFactoryClass()
