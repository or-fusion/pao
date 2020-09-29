import pyomo.environ as pe
import pao.common

#
# TODO - should we have a separate factory for LinearBilevelProblems?
#
LinearBilevelSolver = pao.common.SolverFactory


class LinearBilevelSolverBase(pao.common.Solver):
    """
    Define the API for solvers that optimize a LinearBilevelProblem
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def check_model(self, lbp):         # pragma: no cover
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        lbp.check()

    def solve(self, *args, **kwds):     # pragma: no cover
        #
        # Solve the LinearBilevelProblem
        #
        pass


class LinearBilevelResults(pao.common.Results):

    def __init__(self, solution_manager=None):
        super(pao.common.Results, self).__init__()
        self._solution_manager=solution_manager

    def copy_from_to(self, **kwds):
        self._solution_manager.copy_from_to(**kwds)

    def load_from(self, data):          # pragma: no cover
        assert (False), "load_from() is not implemented"
        self._solution_manager.load_from(data)

