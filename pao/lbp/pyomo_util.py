#
# Utilities for creating Pyomo models using data in
# LinearBilevelProblem objects.
#
import numpy as np
import pyomo.environ as pe
from .repn import LinearLevelRepn, LevelValues, SimplifiedList, LevelVariable


def dot(A, x, num=None):
    print("HERE",type(A), type(x), num)
    if A is None:
        if num is not None:
            if num > 1:
                return np.zeros(num)
            return 0
        return None
    if type(x) is SimplifiedList:
        x = x[0]
    elif type(x) is LevelVariable:
        x = x.pyvar
    if type(A) is SimplifiedList:
        A = A[0]
    elif type(A) is LevelValues:
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
            print(i,j,v)
            e[i] += v*x[j]
        return np.array(e)

def add_variables(block, level):
    pyvar = []
    if level.x.nxR > 0:
        block.xR = pe.Var(range(0,level.x.nxR), within=pe.Reals)
        for i in range(level.x.nxR):
            pyvar.append(block.xR[i])
    else:
        block.xR = None
    if level.x.nxZ > 0:
        block.xZ = pe.Var(range(0,level.x.nxZ), within=pe.Integers)
        for i in range(level.x.nxZ):
            pyvar.append(block.xZ[i])
    else:
        block.xZ = None
    if level.x.nxB > 0:
        block.xB = pe.Var(range(0,level.x.nxB), within=pe.Binaries)
        for i in range(level.x.nxB):
            pyvar.append(block.xB[i])
    else:
        block.xB = None

    level.x.pyvar = np.array(pyvar)

    if level.x.lower_bounds is not None:
        i=0
        while i<level.x.nxR:
            lb = level.x.lower_bounds[i]
            if not lb is np.NINF:
                block.xR[i].setlb( lb )
            i += 1
        while i<level.x.nxZ:
            lb = level.x.lower_bounds[i]
            if not lb is np.NINF:
                block.xZ[i].setlb( lb )
            i += 1
        while i<level.x.nxB:
            lb = level.x.lower_bounds[i]
            block.xB[i].setlb( lb )
            i += 1

    if level.x.upper_bounds is not None:
        i=0
        while i<level.x.nxR:
            ub = level.x.upper_bounds[i]
            if not ub is np.PINF:
                block.xR[i].setub( ub )
            i += 1
        while i<level.x.nxZ:
            ub = level.x.upper_bounds[i]
            if not ub is np.PINF:
                block.xZ[i].setub( ub )
            i += 1
        while i<level.x.nxB:
            ub = level.x.upper_bounds[i]
            block.xB[i].setub( ub )
            i += 1


def add_linear_constraints(block, A, U, L, b, inequalities):
    if b is None:
        return
    nc = b.size
    if nc == 0:
        return
    e = dot(A[U], U.x, num=nc) + dot(A[L], L.x, num=nc)

    print(type(e),type(b))
    print(dot(A[U], U.x, num=nc))
    print("e",e)
    print("b",b)
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
    return dot(A, level.x, num=nc)
    e = np.zeros(nc)
    if len(level.xR) > 0 and A.xR is not None:
        e = e + dot(A.xR, level.xR.var)
    if len(level.xZ) > 0 and A.xZ is not None:
        e = e + dot(A.xZ, level.xZ.var)
    if len(level.xB) > 0 and A.xB is not None:
        e = e + dot(A.xB, level.xB.var)
    return e

def _linear_objective(c, d, U, L, block, minimize):
    e = _linear_expression(1, c[U], U) + _linear_expression(1, c[L], L) + d
    if minimize:
        block.o = pe.Objective(expr=e)
    else:
        block.o = pe.Objective(expr=e, sense=pe.maximize)

def _linear_constraints(inequalities, A, U, L, b, block):
    add_linear_constraints(block, A, U, L, b, inequalities)
