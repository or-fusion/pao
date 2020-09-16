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
        self.name = name

    def check_model(self, lbp):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        lbp.check()

    def solve(self, *args, **kwds):
        #
        # Solve the LinearBilevelProblem
        #
        pass


class LinearBilevelResults(pao.common.Results):

    def __init__(self, solution_manager=None):
        super(pao.common.Results, self).__init__()
        self._solution_manager=solution_manager

    def copy_from_to(self, pyomo_model, lbp):
        self._solution_manager.copy_from_to(pyomo_model, lbp)

