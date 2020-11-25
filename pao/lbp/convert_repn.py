import copy
from scipy.sparse import coo_matrix, dok_matrix, csc_matrix, vstack, hstack
import numpy as np
from .repn import LinearBilevelProblem
from .soln_manager import LBP_SolutionManager

#
# Variable Change objects that cache information needed to
# transform real and integer variables into a non-negative form.
#
class VChange(object):
    def __init__(self, real=True, v=None, cid=None, w=None, lb=None, ub=None):
        self.real = real            # If false, then this is a general integer variable
        self.v = v                  # Index of the current variable, whose coefficient may change
        self.cid = cid
        self.w = w                  # Index of a new variable that needs to be added
        self.lb = lb
        self.ub = ub

    def __str__(self):              # pragma: no cover
        return "VChange(real=%d v=%s cid=%d w=%s lb=%s ub=%s)" % (self.real, str(self.v), self.cid, str(self.w), str(self.lb), str(self.ub))

# Variable with a nonzero lower bound
class VChangeLowerBound(VChange):
    def __init__(self, *, real, v, lb):
        super().__init__(real=real, v=v, cid=1, lb=lb)

# Variable with a finite upper bound
class VChangeUpperBound(VChange):
    def __init__(self, *, real, v, ub):
        super().__init__(real=real, v=v, cid=2, ub=ub)

# Variable with finite lower and upper bounds
class VChangeRange(VChange):
    def __init__(self, *, real, v, lb, ub, w=None):
        super().__init__(real=real, v=v, cid=3, lb=lb, ub=ub, w=w)

# Variable that is unbounded
class VChangeUnbounded(VChange):
    def __init__(self, *, real, v, w):
        super().__init__(real=real, v=v, cid=4, w=w)

class VChanges(object):

    def __init__(self):
        self._data = []
        self.nxR_old = 0
        self.nxZ_old = 0
        self.nxR = 0
        self.nxZ = 0

    def append(self, chg):
        self._data.append(chg)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for chg in self._data:
            yield chg


def _find_nonpositive_variables(V, inequalities):
    changes = VChanges()
    nxV = V.nxR+V.nxZ
    nxR = V.nxR
    nxZ = V.nxZ
    changes.nxR_old = nxR
    changes.nxZ_old = nxZ

    for i in range(nxV):
        lb = V.lower_bounds[i]
        ub = V.upper_bounds[i]
        #print(lb, ub)
        if ub == np.PINF:
            if lb == 0:
                #print("HERE0")
                continue
            elif lb == np.NINF:
                # Unbounded variable
                if i<V.nxR:
                    changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                    #print("HERE1")
                    nxR += 1
                else:
                    changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                    #print("HERE2")
                    nxZ += 1
            else:
                changes.append( VChangeLowerBound(real=i<V.nxR, v=i, lb=lb) )
        elif lb == np.NINF:
            changes.append( VChangeUpperBound(real=i<V.nxR, v=i, ub=ub) )
        elif inequalities:
            changes.append( VChangeRange(real=i<V.nxR, v=i, lb=lb, ub=ub) )
        else:
            changes.append( VChangeRange(real=i<V.nxR, v=i, lb=lb, ub=ub, w=nxR) )
            #print("HERE2")
            nxR += 1

    # Reset the variable id for integers, given the final value of nxR
    for c in changes:
        if c.real is False:
            c.v += nxR-V.nxR
        if type(c) is VChangeUnbounded and c.real is False:
            c.w += nxR

    assert (nxR+nxZ == nxV + sum(1 if c.w is not None else 0 for c in changes))
    changes.nxR = nxR
    changes.nxZ = nxZ
    #print(nxR, nxZ, V.nxB)
    return changes


def _process_changes(changes, V, c, d, A, b, add_rows=False):
    d = copy.copy(d)
    b = copy.copy(b)

    if A is None:
        Acsc = csc_matrix(0)
        nrows = 0
    else:
        Acsc = A.tocsc()
        nrows = A.shape[0]

    B = {}
    for chg in changes:
        v = chg.v
        if type(chg) is VChangeLowerBound:      # real variable bounded below
            lb = chg.lb
            if c is not None:
                d += c[v]*lb
            if A is not None:
                # i is index of the vth column in the A matrix
                i = Acsc.indptr[v]
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    b[row] -= Acsc[row, v]*lb
                    i += 1

        elif type(chg) is VChangeUpperBound:    # real variable bounded above
            ub = chg.ub
            if c is not None:
                d += c[v]*ub
                c[v] *= -1
            if A is not None:
                # i is index of the vth column in the A matrix
                i = Acsc.indptr[v]
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    b[row] -= Acsc[row, v]*ub
                    Acsc[row, v] *= -1
                    i += 1

        elif type(chg) is VChangeRange:         # real variable bounded
            lb = chg.lb
            ub = chg.ub
            w = chg.w
            if c is not None:
                d += c[v]*lb
                if w is not None:
                    c[w] = 0
            if A is not None:
                # i is index of the vth column in the A matrix
                i = Acsc.indptr[v]
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    b[row] -= Acsc[row, v]*lb
                    i += 1
            if add_rows:
                # Add new constraint
                # If w is not None, then we are adding an associated slack variable
                # NOTE: We only add the constraint to the level that "owns" the variables
                b = np.append(b, ub-lb)
                B[nrows, v] = 1
                if w is not None:
                    B[nrows, w] = 1
                nrows += 1

        else:                                   # real variable unbounded
            w = chg.w
            if c is not None:
                c[w] = -c[v]
            if A is not None:
                # i is index of the vth column in the A matrix
                i = Acsc.indptr[v]
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    B[row, w] = - Acsc[row, v]
                    i += 1

    if nrows == 0:
        return c, d, None, b

    Bdok = dok_matrix((nrows, changes.nxR+changes.nxZ+V.nxB))
    #print(Bdok.shape)
    #print(Acsc.shape)
    # Collect the items from B
    for k,v in B.items():
        Bdok[k] = v
    # Merge in the items from A, shifting columns
    Adok = Acsc.todok()
    for k,v in Adok.items():
        Bdok[k] = v
    return c, d, Bdok.tocoo(), b


def convert_to_nonnegative_variables(ans, inequalities):
    #
    # Collect real and integer variables that are changing
    #
    changes = {}
    #ans.print()
    for L in ans.levels():
        changes[L.id] = _find_nonpositive_variables(L.x, inequalities)
        L.resize(nxR=changes[L.id].nxR, nxZ=changes[L.id].nxZ, nxB=L.x.nxB)
        L.x.lower_bounds = np.zeros(len(L.x))

        #print(L.name)
        #for chg in changes[L.id]:
        #    print(chg)
    #
    # Process changes 
    #
    for L in ans.levels():
        if len(changes[L.id]) > 0:
            for X in L.levels():
                X.c[L], X.d, X.A[L], X.b = \
                    _process_changes(changes[L.id], L.x, X.c[L], X.d, X.A[L], X.b, add_rows=L.id == X.id)
    #
    # Resize constraint matrices
    #
    # After processing upper and lower variables, we may have added constraints.  The other
    # upper/lower constraint matrices need to be resized as well.
    #
    #for i in range(len(L)):
    #    if U.A[L[i]] is not None:
    #        U.A[L[i]].resize( [len(U.b), len(L[i].x)] )
    #    if L[i].A[U] is not None:
    #        L[i].A[U].resize( [len(L[i].b), len(U.x)] )
    #
    return changes


def combine_matrices(A, B):         #pragma: nocover
    """
    Combining matrices with different shapes

    Matrix A may be None
    """
    if A is None:
        if B.size > 0:          # pragma: no cover
            return B
        return None

    shape = [max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1])]
    x=A.tocoo()
    y=B.tocoo()
    d = np.concatenate((x.data, y.data))
    r = np.concatenate((x.row, y.row))
    c = np.concatenate((x.col, y.col))
    ans = coo_matrix((d,(r,c)), shape=shape)
    return ans


def convert_sense(L, minimize=True):
    if (minimize and not L.minimize) or (not minimize and L.minimize):
        L.minimize = minimize
        L.d *= -1
        for i in L.c:
            L.c[i] *= -1


def convert_to_minimization(ans):
    for L in ans.levels():
        convert_sense(L, minimize=True)


def add_ineq_constraints(mat):
    x=mat.tocoo()
    nrows = mat.shape[0]
    d = -1 * x.data
    r = x.row #+ nrows
    c = x.col
    newmat = coo_matrix((d,(r,c)), shape=[nrows, mat.shape[1]])
    return vstack([mat, newmat])
    

def convert_constraints(ans, inequalities):
    if inequalities:
        #
        # Creating inequality constraints from equalities by 
        # duplicating constraints
        #
        for L in ans.levels():
            if not L.inequalities:
                bnew = np.copy(L.b)
                bnew *= -1
                L.b = np.concatenate((L.b, bnew))
                for i in L.A:
                    L.A[i] = add_ineq_constraints(L.A[i])
    else:
        #
        # Add slack variables to create equality constraints from inequalities
        #
        for L in ans.levels():
            if L.inequalities and len(L.b) > 0:
                nxR = L.x.nxR
                L.resize( nxR=nxR + len(L.b), nxZ=L.x.nxZ, nxB=L.x.nxB, lb=0 )
                B = L.A[L]
                if B is None:
                    continue
                B = B.todok()
                for i in range(len(L.b)):
                    B[i,nxR+i] = 1
                L.A[L] = B
    #
    # Update inequality values
    #
    for L in ans.levels():
        L.inequalities = inequalities


def get_multipliers(lbp, changes):
    multipliers = {}
    for L in lbp.levels():
        # 
        # If there were no changes, then the multiplier is 1
        #
        multipliers[L.id] =   [[(i,1)] for i in L.x]
        for chg in changes[L.id]:
            if type(chg) is VChangeUpperBound:
                multipliers[L.id][ chg.v ] = [(chg.v,-1)]
            elif type(chg) is VChangeUnbounded:
                multipliers[L.id][ chg.v ] = [(chg.v,1), (chg.w,-1)]
    return multipliers


def convert_binaries_to_integers(lbp):
    for L in lbp.levels():
        if L.x.nxB > 0:
            L.x._resize(nxR=L.x.nxR, nxZ=L.x.nxZ+L.x.nxB, nxB=0, lb=0, ub=1)

"""
            if nxZ == 0:
                lbp.U.c.U.xZ = lbp.U.c.U.xB
                lbp.U.A.U.xZ = lbp.U.A.U.xB
            else:
                lbp.U.c.U.xZ = np.concatenate((lbp.U.c.U.xZ, lbp.U.c.U.xB))
                lbp.U.A.U.xZ = hstack([lbp.U.A.U.xZ, lbp.U.A.U.xB], format='csr')
            lbp.U.c.U.xB = None
            lbp.U.A.U.xB = None
            for i in range(len(lbp.U.L)):
                if nxZ == 0:
                    lbp.U.L[i].c.U.xZ = lbp.U.L[i].c.U.xB
                    lbp.U.L[i].A.U.xZ = lbp.U.L[i].A.U.xB
                else:
                    lbp.U.L[i].c.U.xZ = np.concatenate((lbp.U.L[i].c.U.xZ, lbp.U.L[i].c.U.xB))
                    lbp.U.L[i].A.U.xZ = hstack([lbp.U.L[i].A.U.xZ, lbp.U.L[i].A.U.xB], format='csr')
                lbp.U.L[i].c.U.xB = None
                lbp.U.L[i].A.U.xB = None

    for i in range(len(lbp.U.L)):
        if len(lbp.U.L[i].xB) > 0:
            nxZ = len(lbp.U.L[i].xZ)
            nxB = len(lbp.U.L[i].xB)
            lbp.U.L[i].xZ.resize(nxZ+nxB, lb=0, ub=1)
            lbp.U.L[i].xB.resize(0)

            if nxZ == 0:
                lbp.U.c.U.L[i].xZ = lbp.U.c.U.L[i].xB
                lbp.U.A.U.L[i].xZ = lbp.U.A.U.L[i].xB
            else:
                lbp.U.c.U.L[i].xZ = np.concatenate((lbp.U.c.U.L[i].xZ, lbp.U.c.U.L[i].xB))
                lbp.U.A.L[i].xZ = hstack([lbp.U.A.L[i].xZ, lbp.U.A.L[i].xB], format='csr')
            lbp.U.c.L[i].xB = None
            lbp.U.A.L[i].xB = None
            for i in range(len(lbp.L)):
                if nxZ == 0:
                    lbp.L[i].c.L[i].xZ = lbp.L[i].c.L[i].xB
                    lbp.L[i].A.L[i].xZ = lbp.L[i].A.L[i].xB
                else:
                    lbp.L[i].c.L[i].xZ = np.concatenate((lbp.L[i].c.L[i].xZ, lbp.L[i].c.L[i].xB))
                    lbp.L[i].A.L[i].xZ = hstack([lbp.L[i].A.L[i].xZ, lbp.L[i].A.L[i].xB], format='csr')
                lbp.L[i].c.L[i].xB = None
                lbp.L[i].A.L[i].xB = None
"""

def convert_LinearBilevelProblem_to_standard_form(lbp, inequalities=False):
    """
    After applying this transformation, the problem has the form:
        1. Each real variable x is nonnegative (x >= 0)
        2. Constraints are equalities
    Thus, if a level only has real variables, it will be in standard form
    following this transformation.
    """
    #
    # Clone the LBP object
    #
    ans = lbp.clone()
    #
    # Convert maximization to minimization
    #
    convert_to_minimization(ans)
    #
    # Convert to the required constraint form
    #
    convert_constraints(ans, inequalities)
    #
    # Normalize variables
    #
    changes = convert_to_nonnegative_variables(ans, inequalities)
    #
    # Resize matrices
    #
    for L in ans.levels():
        for X in L.levels():
            A = X.A[L]
            if A is not None:
                A.resize( [len(X.b), len(L.x)] )
    #
    # Setup multipliers that are used to convert variables back to the original model
    #
    multipliers = get_multipliers(lbp, changes)

    return ans, LBP_SolutionManager(multipliers)

