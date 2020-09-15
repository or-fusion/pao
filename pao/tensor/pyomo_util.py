#
# Utilities for creating Pyomo models using data in
# LinearBilevelProblem objects.
#
import numpy as np
import pyomo.environ as pe
from .repn import LinearLevelRepn, LevelValues, SimplifiedList


def dot(A, x, num=None):
    if type(x) is SimplifiedList:
        x = x[0]
    if type(A) is SimplifiedList:
        A = A[0]
    if type(A) is LevelValues:
        assert type(x) is LinearLevelRepn, "Unexpected type %s" % str(type(x))
        if num is not None:
            ans = np.zeros(num)
        else:
            ans = None
        if len(x.xR) > 0 and A.xR is not None:
            if ans is None:
                ans = dot(A.xR, x.xR.var)
            else:
                ans = ans + dot(A.xR, x.xR.var)
        if len(x.xZ) > 0 and A.xZ is not None:
            if ans is None:
                ans = dot(A.xZ, x.xZ.var)
            else:
                ans = ans + dot(A.xZ, x.xZ.var)
        if len(x.xB) > 0 and A.xB is not None:
            if ans is None:
                ans = dot(A.xB, x.xB.var)
            else:
                ans = ans + dot(A.xB, x.xB.var)
        if A._matrix:
            return ans
        if num==1:
            return ans[0]
        return ans

    if type(A) is np.ndarray:
        return sum(A*x)
    else:
        Acoo = A.tocoo()
        e = [0] * Acoo.shape[0]
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
            e[i] += v*x[j]
        return e

def add_variables(block, level):
    if len(level.xR) > 0:
        block.xR = pe.Var(range(0,level.xR.num), within=pe.Reals)
        level.xR.var = np.array([block.xR[i] for i in range(0,level.xR.num)])
        if level.xR.lower_bounds is not None:
            for i,v in block.xR.items():
                v.setlb( level.xR.lower_bounds[i] )
        if level.xR.upper_bounds is not None:
            for i,v in block.xR.items():
                v.setlb( level.xR.upper_bounds[i] )
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

def add_linear_constraints(block, A, U, L, b, inequalities):
    if b is None:
        return
    nc = b.size
    e = dot(A.U, U, num=nc) + dot(A.L, L, num=nc)

    block.c = pe.ConstraintList()
    for i in range(len(e)):
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

#
# Deprecated
#

def _add_upper(*, repn, M_U):
    """
    Add the linear objective and constraints that are associated with the 
    upper-level problem in the LinearBilevelProblem.
    """
    _create_variables(repn.U, M_U)
    _linear_objective(repn.U.c, repn.U, repn.L, repn.U.minimize, M_U)
    _linear_constraints(repn.U.inequalities, repn.U.A, repn.U, repn.L, repn.U.b, M_U)

    fixed = []
    if len(repn.U.xR) > 0:
        fixed.append(M.U.xR)
    if len(repn.U.xZ) > 0:
        fixed.append(M.U.xZ)
    if len(repn.U.xB) > 0:
        fixed.append(M.U.xB)

def _create_variables(level, block):
    add_variables(block, level)

def _linear_expression(nc, A, level):
    return dot(A, level, num=nc)
    e = np.zeros(nc)
    if len(level.xR) > 0 and A.xR is not None:
        e = e + dot(A.xR, level.xR.var)
    if len(level.xZ) > 0 and A.xZ is not None:
        e = e + dot(A.xZ, level.xZ.var)
    if len(level.xB) > 0 and A.xB is not None:
        e = e + dot(A.xB, level.xB.var)
    return e

def _linear_objective(c, d, U, L, block, minimize):
    e = _linear_expression(1, c.U, U) + _linear_expression(1, c.L, L) + d
    if minimize:
        block.o = pe.Objective(expr=e)
    else:
        block.o = pe.Objective(expr=e, sense=pe.maximize)

def _linear_constraints(inequalities, A, U, L, b, block):
    add_linear_constraints(block, A, U, L, b, inequalities)
