#
# Solvers that convert the problem to a LinearBilevelProblem and
# solve using a LBP-specific solver.
#
import time
import pao.mpr
import pao.common
from pao.pyomo.solver import Solver, PyomoSubmodelSolverBase_LBP
from pyomo.common.config import ConfigBlock, ConfigValue


@Solver.register(
        name='pao.pyomo.FA',
        doc="PAO solver for Pyomo models that define linear and bilinear bilevel problems.  Solver uses big-M relaxations discussed by Fortuny-Amat and McCarl (1981).")
class PyomoSubmodelSolver_FA(PyomoSubmodelSolverBase_LBP):
    """
    PAO FA solver for Pyomo models: pao.pyomo.FA

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.mpr.FA solver.
    """

    config = PyomoSubmodelSolverBase_LBP.config()
    config.declare('mip_solver', ConfigValue(
        default='glpk',
        description="The MIP solver used by FA.  (default is glpk)"
        ))
    #config.declare('mip_options', ConfigValue(
    #    default=None,
    #    description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.FA', 'pao.mpr.FA')

    def solve(self, model, **options):
        return super().solve(model, **options)

pao.common.SolverAPI._generate_solve_docstring(PyomoSubmodelSolver_FA)



@Solver.register(
        name='pao.pyomo.MIBS',
        doc="PAO solver for Multilevel Problem Representations using the COIN-OR MibS solver by Tahernejad, Ralphs, and DeNegre (2020).")
class PyomoSubmodelSolver_MIBS(PyomoSubmodelSolverBase_LBP):
    """
    PAO MibS solver for Pyomo models: pao.pyomo.MIBS

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.mpr.MIBS solver.
    """

    config = PyomoSubmodelSolverBase_LBP.config()
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.MIBS', 'pao.mpr.MIBS')

    def solve(self, model, **options):
        return super().solve(model, **options)

pao.common.SolverAPI._generate_solve_docstring(PyomoSubmodelSolver_MIBS)


@Solver.register(
        name='pao.pyomo.REG',
        doc="PAO solver for Pyomo models that define linear and bilinear bilevel problems.  Solver uses regularization discussed by Scheel and Scholtes (2000) and Ralph and Wright (2004).")
class PyomoSubmodelSolver_REG(PyomoSubmodelSolverBase_LBP):
    """
    PAO REG solver for Pyomo models: pao.pyomo.REG

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.mpr.REG solver.
    """

    config = PyomoSubmodelSolverBase_LBP.config()
    config.declare('nlp_solver', ConfigValue(
        default='ipopt',
        description="The NLP solver used by REG.  (default is ipopt)"
        ))
    #config.declare('nlp_options', ConfigValue(
    #    default=None,
    #    description="A dictionary that defines the solver options for the NLP solver.  (default is None)"))
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.REG', 'pao.mpr.REG')

    def solve(self, model, **options):
        return super().solve(model, **options)

pao.common.SolverAPI._generate_solve_docstring(PyomoSubmodelSolver_REG)


@Solver.register(
        name='pao.pyomo.PCCG',
        doc="PAO solver for Pyomo models that define linear and bilinear bilevel problems.  Solver uses projected column constraint generation algorithm described by Yue et al. (2017)")
class PyomoSubmodelSolver_PCCG(PyomoSubmodelSolverBase_LBP):
    """
    PAO PCCG solver for Pyomo models: pao.pyomo.PCCG

    This solver converts the Pyomo model to a LinearBilevelProblem and
    calls the pao.mpr.PCCG solver.
    """

    config = PyomoSubmodelSolverBase_LBP.config()
    config.declare('mip_solver', ConfigValue(
        default='cbc',
        description="The MIP solver used by PCCG.  (default is cbc)"
        ))
    #config.declare('mip_options', ConfigValue(
    #    default=None,
    #    description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
    
    def __init__(self, **kwds):
        super().__init__('pao.pyomo.PCCG', 'pao.mpr.PCCG')

    def solve(self, model, **options):
        return super().solve(model, **options)

pao.common.SolverAPI._generate_solve_docstring(PyomoSubmodelSolver_PCCG)


