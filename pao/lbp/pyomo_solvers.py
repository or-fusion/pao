#import numpy as np
import pyomo.environ as pe
import pao.bilevel
#import pao.bilevel.plugins
from .solver import SolverFactory
from .repn import LinearBilevelProblem
from . import pyomo_util


class PyomoSolverBase(object):

    def _create_constraints(self, repn, M):
        pass

    def create_pyomo_model(self, repn):
        M = pe.ConcreteModel()
        self.model = M
        self.repn = repn
        #
        # Variables
        #
        M.U = pe.Block()
        pyomo_util._create_variables(repn.U, M.U)
        fixed = []
        if repn.U.x.nxR > 0:
            fixed.append(M.U.xR)
        if repn.U.x.nxZ > 0:
            fixed.append(M.U.xZ)
        if repn.U.x.nxB > 0:
            fixed.append(M.U.xB)
        M.L = pao.bilevel.SubModel(fixed=fixed)
        pyomo_util._create_variables(repn.U.LL, M.L)
        #
        # Objectives
        #
        self._create_objectives(repn, M)
        #
        # Constraints
        #
        self._create_constraints(repn, M)

    def _collect_values(self, level, block):
        if level.x.nxR > 0:
            for i,v in block.xR.items():
                level.x.values[i] = pe.value(v)
        if level.x.nxZ > 0:
            for i,v in block.xZ.items():
                level.x.values[i+level.x.nxR] = pe.value(v)
        if level.x.nxB > 0:
            for i,v in block.xB.items():
                level.x.values[i+level.x.nxR+level.x.nxZ] = pe.value(v)

    def collect_values(self):
        self._collect_values(self.repn.U, self.model.U)
        self._collect_values(self.repn.U.LL, self.model.L)


class PyomoSolverBase_LinearBilevelProblem(PyomoSolverBase):

    def _create_objectives(self, repn, M):
        U = repn.U
        L = repn.U.LL[0]
        pyomo_util._linear_objective(U.c, U.d, U, L, M.U, U.minimize)
        pyomo_util._linear_objective(L.c, L.d, U, L, M.L, L.minimize)
        
    def _create_constraints(self, repn, M):
        U = repn.U
        L = repn.U.LL[0]
        pyomo_util._linear_constraints(U.inequalities, U.A, U, L, U.b, M.U)
        pyomo_util._linear_constraints(L.inequalities, L.A, U, L, L.b, M.L)
        
    def create_pyomo_model(self, repn):
        PyomoSolverBase.create_pyomo_model(self,repn)


#
# For now, we call solvers in pao.bilevel
#

class BilevelSolver1_LinearBilevelProblem(PyomoSolverBase_LinearBilevelProblem):

    def __init__(self, **kwds):
        self.solver_type = 'pao.bilevel.ld'
        self.solver = pe.SolverFactory('pao.bilevel.ld', **kwds)

    def check_model(self, M):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(M) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.solver_type
        M.check()
        #
        # No binary or integer lower level variables
        #
        assert (len(M.L.xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
        assert (len(M.L.xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type
        #
        # Upper and lower objectives are the opposite of each other
        #

    def solve(self, *args, **kwds):
        self.check_model(args[0])
        self.create_pyomo_model(args[0])
        #self.model.pprint()
        newargs = [self.model]
        self.solver.solve(*newargs, **kwds)
        self.collect_values()


class BilevelSolver2_LinearBilevelProblem(PyomoSolverBase_LinearBilevelProblem):

    def __init__(self, **kwds):
        self.solver_type = 'pao.bilevel.blp_global'
        self.solver = pe.SolverFactory('pao.bilevel.blp_global', **kwds)

    def check_model(self, M):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(M) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.solver_type
        M.check()
        #
        # Confirm that we only have one lower-level problem (for now)
        #
        assert (len(M.U.LL) == 1), "Solver '%s' can only solve a LinearBilevelProblem with one lower-level problem" % self.solver_type
        assert (len(M.U.LL[0].LL) == 0), "Solver '%s' can only solve a LinearBilevelProblem that is bilevel" % self.solver_type
        #
        # No binary or integer lower level variables
        #
        for L in M.U.LL:
            assert (L.x.nxZ == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
            assert (L.x.nxB == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type
        #
        # Upper and lower objectives are the opposite of each other
        #

    def solve(self, *args, **kwds):
        self.check_model(args[0])
        self.create_pyomo_model(args[0])
        newargs = [self.model]
        self.solver.solve(*newargs, **kwds)
        self.collect_values()


class BilevelSolver3_LinearBilevelProblem(PyomoSolverBase_LinearBilevelProblem):

    def __init__(self, **kwds):
        self.solver_type = 'pao.bilevel.blp_local'
        self.solver = pe.SolverFactory('pao.bilevel.blp_local', **kwds)

    def check_model(self, M):
        #
        # Confirm that the LinearBilevelProblem is well-formed
        #
        assert (type(M) is LinearBilevelProblem), "Solver '%s' can only solve a LinearBilevelProblem" % self.solver_type
        M.check()
        #
        # Confirm that we only have one lower-level problem (for now)
        #
        assert (len(M.L) == 1), "Solver '%s' can only solve a LinearBilevelProblem with one lower-level problem" % self.solver_type
        #
        # No binary or integer lower level variables
        #
        assert (len(M.L.xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
        assert (len(M.L.xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type
        #
        # Upper and lower objectives are the opposite of each other
        #

    def solve(self, *args, **kwds):
        self.check_model(args[0])
        self.create_pyomo_model(args[0])
        newargs = [self.model]
        self.solver.solve(*newargs, **kwds)
        self.collect_values()


SolverFactory.register(BilevelSolver1_LinearBilevelProblem, name='pao.bilevel.ld')
SolverFactory.register(BilevelSolver2_LinearBilevelProblem, name='pao.bilevel.blp_global')
SolverFactory.register(BilevelSolver3_LinearBilevelProblem, name='pao.bilevel.blp_local')

