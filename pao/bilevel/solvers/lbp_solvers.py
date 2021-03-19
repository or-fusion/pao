#
# Solvers that convert the problem to a LinearBilevelProblem and
# solve using a LBP-specific solver.
#
import time
import pao.lbp
import pao.common
from pao.bilevel.solver import SolverFactory, PyomoSubmodelSolverBase_LBP
from pyomo.common.config import ConfigBlock, ConfigValue


@SolverFactory.register(
        name='pao.pyomo.FA',
        doc=SolverFactory.doc('pao.lbp.FA'))
class PyomoSubmodelSolver_FA(PyomoSubmodelSolverBase_LBP):
    """
    PAO FA solver for Pyomo models: pao.pyomo.FA.

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.lbp.FA solver.
    """

    config = pao.common.solver.SolverAPI.config()
    config.declare('solver', ConfigValue(
        default='glpk',
        description="The name of the MIP solver used by FA.  (default is glpk)"
        ))
    config.declare('solver_options', ConfigValue(
        default=None,
        description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.FA', 'pao.lbp.FA')

    def solve(self, model, **options):
        return super().solve(model, **options)

PyomoSubmodelSolver_FA._update_solve_docstring(PyomoSubmodelSolver_FA.config)



@SolverFactory.register(
        name='pao.pyomo.REG',
        doc=SolverFactory.doc('pao.lbp.REG'))
class PyomoSubmodelSolver_REG(PyomoSubmodelSolverBase_LBP):
    """
    PAO REG solver for Pyomo models: pao.pyomo.REG.

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.lbp.REG solver.
    """

    config = pao.common.solver.SolverAPI.config()
    config.declare('solver', ConfigValue(
        default='ipopt',
        description="The name of the NLP solver used by REG.  (default is ipopt)"
        ))
    config.declare('solver_options', ConfigValue(
        default=None,
        description="A dictionary that defines the solver options for the NLP solver.  (default is None)"))
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.REG', 'pao.lbp.REG')

    def solve(self, model, **options):
        return super().solve(model, **options)

PyomoSubmodelSolver_REG._update_solve_docstring(PyomoSubmodelSolver_REG.config)


@SolverFactory.register(
        name='pao.pyomo.PCCG',
        doc=SolverFactory.doc('pao.lbp.PCCG'))
class PyomoSubmodelSolver_PCCG(PyomoSubmodelSolverBase_LBP):
    """
    PAO PCCG solver for Pyomo models: pao.pyomo.PCCG.

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.lbp.PCCG solver.
    """

    config = pao.common.solver.SolverAPI.config()
    config.declare('solver', ConfigValue(
        default='cbc',
        description="The name of the MIP solver used by PCCG.  (default is cbc)"
        ))
    config.declare('solver_options', ConfigValue(
        default=None,
        description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.PCCG', 'pao.lbp.PCCG')

    def solve(self, model, **options):
        return super().solve(model, **options)

PyomoSubmodelSolver_PCCG._update_solve_docstring(PyomoSubmodelSolver_PCCG.config)


