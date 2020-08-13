import numpy as np
import pyomo.environ as pe
import pao.bilevel
import pao.bilevel.plugins
from .solver import register_solver
from .repn import LinearBilevelProblem


def dot(A, x):
    if type(A) is np.ndarray:
        return A*x
    else: 
        Acoo = A.tocoo()    
        e = [0] * Acoo.shape[0]
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
            e[i] += v*x[j]
        return e

class PyomoSolverBase(object):

    def _create_variables(self, level, block):
        if len(level.xR) > 0:
            block.xR = pe.Var(range(0,level.xR.num), within=pe.Reals)
            level.xR.var = np.array([block.xR[i] for i in range(0,level.xR.num)])
            if level.xR.lower_bounds:
                for i,v in block.xR.items():
                    v.lb = level.xR.lower_bounds[i]
            if level.xR.upper_bounds:
                for i,v in block.xR.items():
                    v.lb = level.xR.upper_bounds[i]
        if len(level.xZ) > 0:
            block.xZ = pe.Var(range(0,level.xZ.num), within=pe.Integers)
            level.xZ.var = np.array([block.xZ[i] for i in range(0,level.xZ.num)])
            if level.xZ.lower_bounds:
                for i,v in block.xZ.items():
                    v.lb = level.xZ.lower_bounds[i]
            if level.xZ.upper_bounds:
                for i,v in block.xZ.items():
                    v.lb = level.xZ.upper_bounds[i]
        if len(level.xB) > 0:
            block.xB = pe.Var(range(0,level.xB.num), within=pe.Binaries)
            level.xB.var = np.array([block.xB[i] for i in range(0,level.xB.num)])

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
        self._create_variables(repn.U, M.U)
        fixed = []
        if len(repn.U.xR) > 0:
            fixed.append(M.U.xR)
        if len(repn.U.xZ) > 0:
            fixed.append(M.U.xZ)
        if len(repn.U.xB) > 0:
            fixed.append(M.U.xB)
        M.L = pao.bilevel.SubModel(fixed=fixed)
        self._create_variables(repn.L, M.L)
        #
        # Objectives
        #
        self._create_objectives(repn, M)
        #
        # Constraints
        #
        self._create_constraints(repn, M)

    def _collect_values(self, level, block):
        if len(level.xR) > 0:
            for i,v in block.xR.items():
                level.xR.values[i] = pe.value(v)
        if len(level.xZ) > 0:
            for i,v in block.xZ.items():
                level.xZ.values[i] = pe.value(v)
        if len(level.xB) > 0:
            for i,v in block.xB.items():
                level.xB.values[i] = pe.value(v)

    def collect_values(self):
        self._collect_values(self.repn.U, self.model.U)
        self._collect_values(self.repn.L, self.model.L)


class PyomoSolverBase_LinearBilevelProblem(PyomoSolverBase):

    def _linear_expression(self, nc, A, level):
        e = np.zeros(nc)
        if len(level.xR) > 0 and A.xR is not None:
            e = e + dot(A.xR, level.xR.var)
        if len(level.xZ) > 0 and A.xZ is not None:
            e = e + dot(A.xZ, level.xZ.var)
        if len(level.xB) > 0 and A.xB is not None:
            e = e + dot(A.xB, level.xB.var)
        return e

    def _linear_objective(self, c, U, L, block, minimize):
        e = self._linear_expression(1, c.U, U) + self._linear_expression(1, c.L, L)
        if minimize:
            block.o = pe.Objective(expr=e[0])
        else:
            block.o = pe.Objective(expr=e[0], sense=pe.maximize)

    def _create_objectives(self, repn, M):
        self._linear_objective(repn.U.c, repn.U, repn.L, M.U, repn.U.minimize)
        self._linear_objective(repn.L.c, repn.U, repn.L, M.L, repn.L.minimize)
        
    def _linear_constraints(self, inequalities, A, U, L, b, block):
        if b is None:
            return
        nc = b.size
        e = self._linear_expression(nc, A.U, U) + self._linear_expression(nc, A.L, L)

        block.c = pe.ConstraintList()
        for i in range(e.size):
            if type(e[i]) in [int,float]:
                if inequalities:
                    assert e[i] <= b[i], "Trivial linear constraint violated: %f <= %f" % (e[i], b[i])
                else:
                    assert e[i] == b[i], "Trivial linear constraint violated: %f == %f" % (e[i], b[i])
                continue
            if inequalities:
                block.c.add( e[i] <= b[i] )
            else:
                block.c.add( e[i] == b[i] )

    def _create_constraints(self, repn, M):
        self._linear_constraints(repn.U.inequalities, repn.U.A, repn.U, repn.L, repn.U.b, M.U)
        self._linear_constraints(repn.L.inequalities, repn.L.A, repn.U, repn.L, repn.L.b, M.L)
        
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
        assert (len(M.L) == 1), "Solver '%s' can only solve a LinearBilevelProblem with one lower-level problem" % self.solver_type
        #
        # No binary or integer lower level variables
        #
        for L in M.L:
            assert (len(L.xZ) == 0), "Cannot use solver %s with model with integer lower-level variables" % self.solver_type
            assert (len(L.xB) == 0), "Cannot use solver %s with model with binary lower-level variables" % self.solver_type
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


register_solver('pao.bilevel.ld', BilevelSolver1_LinearBilevelProblem)
register_solver('pao.bilevel.blp_global', BilevelSolver2_LinearBilevelProblem)
register_solver('pao.bilevel.blp_local', BilevelSolver3_LinearBilevelProblem)

