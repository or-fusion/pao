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
from munch import Munch
import pyutilib
import pyomo.environ as pe
import pyomo.opt
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.mpec import ComplementarityList, complements

import pao.common
from ..solver import Solver, LinearMultilevelSolverBase, LinearMultilevelResults
from ..repn import LinearMultilevelProblem
from ..convert_repn import convert_to_standard_form, convert_sense, convert_binaries_to_integers
from . import pyomo_util
from .pccg_solver import execute_PCCG_solver


@Solver.register(
        name='pao.mpr.PCCG',
        doc='PAO solver for Multilevel Problem Representations that define linear bilevel problems. Solver uses projected column constraint generation algorithm described by Yue et al. (2017).')
class LinearMultilevelSolver_PCCG(LinearMultilevelSolverBase):
    """
    PAO PCCG solver for linear MPRs: pao.mpr.PCCG

    This solver iteratively adds constraints to tighten a relaxation of the lower-level problem.
    """

    config = LinearMultilevelSolverBase.config()
    config.declare('mip_solver', ConfigValue(
        default='cbc',
        description="The MIP solver used by PCCG.  (default is cbc)"
        ))
    #config.declare('mip_options', ConfigValue(
    #    default=None,
    #    description="A dictionary that defines the solver options for the MIP solver.  (default is None)"))
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
        description="Convergence tolerance for \|UB-LB\|. (default is 1e-8)"
        ))
    config.declare('rtol', ConfigValue(
        default=1e-8,
        domain=float,
        description="Convergence tolerance for \|UB-LB\|. (default is 1e-8)"
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
        super().__init__(name='pao.mpr.PCCG')

    def check_model(self, mpr):
        #
        # Confirm that the LinearMultilevelProblem is well-formed
        #
        assert (type(mpr) is LinearMultilevelProblem), "Solver '%s' can only solve a LinearMultilevelProblem" % self.name
        mpr.check()
        #
        assert (len(mpr.U.LL) == 1), "Can only solve linear bilevel problems with one lower-level"
        #
        assert (len(mpr.U.LL.LL) == 0), "Can only solve bilevel problems"

    def solve(self, mpr, options=None, **config_options):
        #
        # Error checks
        #
        self.check_model(mpr)
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
        self.standard_form, soln_manager = convert_to_standard_form(mpr, inequalities=True)
        convert_sense(self.standard_form.U.LL, minimize=False)
        convert_binaries_to_integers(self.standard_form)
        
        results = LinearMultilevelResults(solution_manager=soln_manager)

        UxR, UxZ, LxR, LxZ = execute_PCCG_solver(self.standard_form, self.config, results)
        xR = {mpr.U.id:UxR, mpr.U.LL[0].id:LxR}
        xZ = {mpr.U.id:UxZ, mpr.U.LL[0].id:LxZ}

        if False:
            print("UxR")
            for i in UxR:
                print(i, UxR[i].value)
            print("UxZ")
            for i in UxZ:
                print(i, UxZ[i].value)
            print("LxR")
            for i in LxR:
                print(i, LxR[i].value)
            print("LxZ")
            for i in LxZ:
                print(i, LxZ[i].value)

        results.copy_solution(From=Munch(LxR=xR, LxZ=xZ), To=mpr)

        results.solver.wallclock_time = time.time() - start_time
        return results


pao.common.SolverAPI._generate_solve_docstring(LinearMultilevelSolver_PCCG)
