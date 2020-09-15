import pyomo.environ as pe


solver_registry = {}

def register_solver(name, cls):
    solver_registry[name] = cls


def LinearBilevelSolver(name):
    assert (name in solver_registry), "Unknown solver '%s' specified" % name
    return solver_registry[name]()


class LinearBilevelSolverBase(object):
    """
    Define the API for solvers that optimize a LinearBilevelProblem
    """

    def __init__(self, name):
        self.name = 'unknown'

    def check_model(self, M):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        pass

    def solve(self, *args, **kwds):
        #
        # Solve the LinearBilevelProblem
        #
        pass
