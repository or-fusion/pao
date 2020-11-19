#
# Utilities for creating Pyomo models using data in
# LinearBilevelProblem objects.
#
import numpy as np
import pyomo.environ as pe
from .repn import LinearLevelRepn, LevelValues, SimplifiedList, LevelVariable


def dot(A, x, num=None):
    if A is None:
        if num is not None:
            if num > 1:
                return np.zeros(num)
            return 0
        return None
    #if type(x) is SimplifiedList:
    #    x = x[0]
    if type(x) is LevelVariable:
        x = x.pyvar
    #if type(A) is SimplifiedList:
    #    A = A[0]

    if type(A) is np.ndarray:
        return sum(A*x)
    else:
        Acoo = A.tocoo()
        e = [0] * Acoo.shape[0]
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
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
    assert (b is not None), "Unexpected 'None' value for constraint RHS"
    nc = b.size
    if nc == 0:
        return
    e = dot(A[U], U.x, num=nc) + dot(A[L], L.x, num=nc)

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

