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
        super(pao.common.Solver, self).__init__()
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
