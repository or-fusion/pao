#
# Classes used to define a solver API
#
import copy
import six
import abc
import enum
import textwrap
from pyutilib.misc import Options
from pyomo.common.config import ConfigValue, ConfigBlock, add_docstring_list

__all__ = ['TerminationCondition', 'SolverAPI', 'Results', 'Solver']


class TerminationCondition(enum.Enum):
    """
    The TerminationCondition class defines a enumeration of optimization termination conditions.
    """

    unknown = 0
    """unknown serves as both a default value, and it is used when no other enum member makes sense"""

    maxTimeLimit = 1
    """The solver exited due to a time limit"""

    maxIterations = 2
    """The solver exited due to an iteration limit """

    objectiveLimit = 3
    """The solver exited due to an objective limit"""

    minStepLength = 4
    """The solver exited due to a minimum step length"""

    optimal = 5
    """The solver exited with an optimal solution"""

    unbounded = 8
    """The solver exited because the problem is unbounded"""

    infeasible = 9
    """The solver exited because the problem is infeasible"""

    infeasibleOrUnbounded = 10
    """The solver exited because the problem is either infeasible or unbounded"""

    error = 11
    """The solver exited due to an error"""

    interrupted = 12
    """The solver exited because it was interrupted"""

    licensingProblems = 13
    """The solver exited due to licensing problems"""


class SolverAPI(abc.ABC):
    """
    The base class for all PAO solvers.

    The SolverAPI class defines a consistent API for optimization solvers.
    """

    config = ConfigBlock()
    config.declare('time_limit', ConfigValue(
        default=None,
        domain=float,
        description="Solver time limit (in seconds)",
        ))
    config.declare('keep_files', ConfigValue(
        default=False,
        domain=bool,
        description="If True, then temporary files are not deleted. (default is False)",
        ))
    config.declare('tee', ConfigValue(
        default=False,
        domain=bool,
        description="If True, then solver output is streamed to stdout. (default is False)",
        ))
    config.declare('load_solutions', ConfigValue(
        default=True,
        domain=bool,
        description="If True, then the finale solution is loaded into the model. (default is True)",
        ))

    def __init__(self):
        # Create a per-instance copy of the configuration data
        self.config = self.config()

    def available(self):
        """
        Returns a bool indicating if the solver can be executed.

        The default behavior is to always return `True`, but this method
        can be overloaded in a subclass to support solver-specific logic
        (e.g.  to confirm that a solver license is available).

        Returns
        -------
        bool
            This method returns True if the solver can be executed.
        """
        True

    def valid_license(self):
        """
        Returns a bool indicating if the solver has a valid license.

        The default behavior is to always return `True`, but this method
        can be overloaded in a subclass to support solver-specific logic
        (e.g.  to check the solver license).

        Returns
        -------
        bool
            This method returns True if the solver license is valid.
        """
        True

    def version(self):
        """
        Returns a tuple that describes the solver version.

        The return value is a tuple of strings.  A typical format is (major, minor, patch), but this
        is not required. The default behavior is to return an empty tuple.

        Returns
        -------
        tuple
            The solver version.
        """
        return tuple()

    @abc.abstractmethod
    def solve(self, model, **options):
        """
        Executes the solver and loads the solution into the model.

        Parameters
        ----------
        model
            The model that is being optimized.
        options
            Keyword options that are used to configure the solver.

        {}
        Returns
        -------
        Results
            A summary of the optimization results.
        """
        pass

    __solve_doc__ = solve.__doc__

    @staticmethod
    def _generate_solve_docstring(cls):
        """
        Generate the docstring for cls.solve, including the description
        of the keyword arguments defined by a pyomo configuration object.

        Parameters
        ----------
        cls:
            The subclass of SolverAPI whose docstring is being generated
        """
        cls.solve.__doc__ = SolverAPI.__solve_doc__.format( add_docstring_list("", cls.config, 8) )

    def is_persistent():
        """
        Returns True if the solver is persistent.

        Persistent solvers maintain the model representation in memory,
        which enables performance optimization when a problem is resolved
        after changing initial conditions or tweaking model parameters.

        The default is to return False, but this method can be overloaded
        in a subclass to support solver-specific logic.

        Returns
        -------
        bool
            This method returns True if the solver is persistent.
        """
        return False

    def __bool__(self):
        """
        Raises an error because this class cannot be interpreted with a boolean.

        Raises
        ------
        RuntimeError
            Casting a solver to bool is not allowed.
        """
        raise RuntimeError("Casting a solver to bool() is not allowed.  Use available() to check if the solver can be executed.")

    def __enter__(self):
        """
        Setup a solver and return it within a context manager.

        Returns
        -------
        SolverAPI
            Return a reference to **self**.
        """
        return self

    def __exit__(self, t, v, traceback):
        """
        Cleanup the solver at the end of a context manager.
        """
        pass

    def _update_config(self, config_options, validate_options=True):
        """
        .. todo::
            Document where and how this is used.
        """
        keys = set(config_options.keys())
        for k,v in config_options.items():
            if k in self.config:
                self.config[k] = v
                keys.remove(k)    
        if validate_options:
            assert (len(keys) == 0), "Unexpected options to solve() have been specified: %s" % " ".join(sorted(k for k in keys))
        return {key:config_options[key] for key in keys}

SolverAPI._generate_solve_docstring(SolverAPI)


"""
>>> opt = Solver('my_solver')
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
class ResultsBase(abc.ABC):
    """
    Defines the API for results objects.

    A Results object is returned from this method.  This object
    reports information from a solver's execution, including the
    termination condition, objective value for the best feasible point
    and the best bound computed on the objective value.  Additionally,
    a results object may contain solutions found during optimization.

    Attributes
    ----------
    solution_manager
        An object that manages storing and loading data to and from models.
    """

    def __init__(self):
        self.solution_manager = None
        self.solver = Options(
                termination_condition=TerminationCondition.unknown,
                best_feasible_objective=None,
                )
        self.problem = Options()

    @abc.abstractmethod
    def found_feasible_solution(self):
        """
        Returns
        -------
        bool
            True if at least one feasible solution was found. False otherwise.
        """
        pass

    def store_to(self, model, i=0):
        """
        Store the solution in this object into the given model.

        A results object may contain one or more solutions. This method
        copies the **i**-th solution into the model.

        Parameters
        ----------
        i: int
            The index of the solution copied into the model.
        """
        if self.solution_manager is None:
            raise RuntimeError("The results object is missing a solution manager.")
        self.solution_manager.store_to(model, i=0)

    def load_from(self, model):
        """
        Load solution from the model.

        When completed, this results object contains only one solution, which corresponds
        to the solution in the model.
        """
        if self.solution_manager is None:
            raise RuntimeError("The results object is missing a solution manager.")
        self.solution_manager.load_from(model)

    @abc.abstractmethod
    def __str__(self):
        """
        Generate a string summary of this results object.

        Returns
        -------
        str
            A string summarizing the data in this object.
        """
        pass


class Results(ResultsBase):
    """
    The results object.
    """

    def __init__(self, found_feasible_solution=None):
        super(Results, self).__init__()
        self._found_feasible_solution = found_feasible_solution

    def found_feasible_solution(self):
        return self._found_feasible_solution

    def __str__(self):
        problem = self.problem.__str__(indent='  ')
        solver = self.solver.__str__(indent='  ')
        return "Problem:\n-" + problem[1:] + \
            "\nSolver:\n-" + solver[1:]


class SolverFactory(object):
    """
    A class that manages a registry of solvers.

    A solver factory manages a registry that enables 
    solvers to be created by name.
    """

    _registry = {}
    _doc = {}

    def register(self, cls=None, *, name=None, doc=None):
        """
        Register a solver with the specified name.

        Parameters
        ----------
        cls
            Class type for the solver
        name: str
            Unique name of the solver
        doc: str
            Short description of the solver

        Returns
        -------
        decorator
            If the **cls** parameter is None, then a class 
            decorator function
            is returned that can be used to register a solver.
        """
        def decorator(cls):
            assert (name is not None), "Must register a solver with a name"
            #assert (name in SolverFactory._registry), "Name '%s' is already registered!" % name
            SolverFactory._registry[name] = cls
            SolverFactory._doc[name] = doc
            return cls
        if cls is None:
            return decorator
        return decorator(cls)

    def __iter__(self):
        """
        Returns
        -------
        generator
            Returns an iterator that generates the solver names.
        """
        for name in sorted(SolverFactory._doc.keys()):
            yield name

    def summary(self):
        """
        Print a summary of all solvers.
        """
        for name in self:
            print(name)
            print(textwrap.indent("\n".join(textwrap.wrap(self.description(name))), "    "))
            print("")

    def description(self, name):
        """
        Returns the description of the specified solver.

        Parameters
        ----------
        name: str
            The name of a solver

        Returns
        -------
        str
            A short description of the specified solver
        """
        assert (name in SolverFactory._registry), "Unknown solver '%s' specified" % name
        return SolverFactory._doc[name]

    def __call__(self, name):
        """
        Constructs the specified solver.

        This method creates a class instance for the solver that is specified.

        Parameters
        ----------
        name: str
            The name of a solver

        Returns
        -------
        SolverAPI
            A solver class instance for the solver that is specified.
        """
        assert (name in SolverFactory._registry), "Unknown solver '%s' specified" % name
        return SolverFactory._registry[name]()


Solver = SolverFactory()
"""
Solver is a global instance of the :class:`SolverFactory`.
This object provides a global registry of PAO optimization solvers.
"""
