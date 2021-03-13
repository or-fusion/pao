#
# A solver for linear bilevel programs using
# using projected column constraint generation
# "A projection-based reformulation and decomposition algorithm for global optimization 
#  of a class of mixed integer bilevel linear programs" by Dajun Yue, Jiyao Gao, Bo Zeng, Fengqi You
#
# Adapted from an implementation by She'ifa Punla-Green at Sandia National Labs
#
#This algorithm seeks to solve the following bilevel MILP:
#    min cR*xu + cZ*yu + dR*xl0 + dZ*yl0 
#    s.t. AR*xu + AZ*yu + BR*xl0 + BZ* yl0 <= r
#     (xl0,yl0) in argmax {wR*xl+wZ*yl: PR*xl+PZ*yl<=s-QR*xu-QZ*yu}
#
import time
import numpy as np
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.mpec import ComplementarityList, complements
from ..solver import SolverFactory, LinearMultilevelSolverBase, LinearMultilevelResults
from ..repn import LinearMultilevelProblem
from ..convert_repn import convert_to_standard_form, convert_sense, convert_binaries_to_integers
from . import pyomo_util
from .pccg_solver import execute_PCCG_solver


@SolverFactory.register(
        name='pao.lbp.PCCG',
        doc='A solver for linear bilevel programs using using projected column constraint generation')
class LinearMultilevelSolver_PCCG(LinearMultilevelSolverBase):

    config = LinearMultilevelSolverBase.config()
    config.declare('solver', ConfigValue(
        default='cbc',
        description="The name of the MIP solver used by PCCG.  (default is cbc)"
        ))
    config.declare('solver_options', ConfigValue(
        default=None,
        description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
    config.declare('bigm', ConfigValue(
        default=1e6,
        domain=float,
        description="The big-M value used to enforce complementarity conditions.       (default is 1e6)"
        ))
    config.declare('epsilon', ConfigValue(
        default=1e-4,
        domain=float,
        description="Parameter used in disjunction approximation. (default is 1e-4)"
        ))
    config.declare('atol', ConfigValue(
        default=1e-8,
        domain=float,
        description="Convergence tolerance for |UB-LB|. (default is 1e-8)"
        ))
    config.declare('rtol', ConfigValue(
        default=1e-8,
        domain=float,
        description="Convergence tolerance for |UB-LB|. (default is 1e-8)"
        ))
    config.declare('maxit', ConfigValue(
        default=None,
        domain=int,
        description="Maximum number of iterations. (default is None)"
        ))
    config.declare('quiet', ConfigValue(
        default=True,
        domain=bool,
        description="If False, then enable verbose solver output. (default is True)"
        ))

    def __init__(self, **kwds):
        super().__init__(name='pao.lbp.PCCG')

    def check_model(self, lbp):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(lbp) is LinearMultilevelProblem), "Solver '%s' can only solve a LinearMultilevelProblem" % self.name
        lbp.check()
        #
        assert (len(lbp.U.LL) == 1), "Can only solve linear bilevel problems with one lower-level"
        #
        assert (len(lbp.U.LL.LL) == 0), "Can only solve bilevel problems"

    def solve(self, lbp, options=None, **config_options):
        #
        # Error checks
        #
        self.check_model(lbp)
        #
        # Process keyword options
        #
        self._update_config(config_options)
        #
        # Start clock
        #
        start_time = time.time()

        # PCCG requires a standard form with inequalities and 
        # a maximization lower-level
        self.standard_form, soln_manager = convert_to_standard_form(lbp, inequalities=True)
        convert_sense(self.standard_form.U.LL, minimize=False)
        
        results = LinearMultilevelResults(solution_manager=soln_manager)

        UxR, UxZ, LxR, LxZ = execute_PCCG_solver(self.standard_form, self.config, results)
        xR = {lbp.U.id:UxR, lbp.U.LL[0].id:LxR}
        xZ = {lbp.U.id:UxZ, lbp.U.LL[0].id:LxZ}
        results.copy_from_to(LxR=xR, LxZ=xZ, lbp=lbp)

        results.solver.wallclock_time = time.time() - start_time
        return results


LinearMultilevelSolver_PCCG._update_solve_docstring(LinearMultilevelSolver_PCCG.config)
