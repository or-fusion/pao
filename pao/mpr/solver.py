import pyomo.environ as pe
import pao.common

Solver = pao.common.Solver


class LinearMultilevelSolverBase(pao.common.SolverAPI):
    """
    Define the API for solvers that optimize a LinearMultilevelProblem
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def check_model(self, lmp):         # pragma: no cover
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        lmp.check()

    def solve(self, *args, **kwds):     # pragma: no cover
        #
        # Solve the LinearMultilevelProblem
        #
        pass


class LinearMultilevelResults(pao.common.Results):

    def __init__(self, solution_manager=None):
        super(pao.common.Results, self).__init__()
        self._solution_manager=solution_manager

    def copy_solution(self, **kwds):
        self._solution_manager.copy(**kwds)

    def load_from(self, data):          # pragma: no cover
        assert (False), "load_from() is not implemented"
        self._solution_manager.load_from(data)

