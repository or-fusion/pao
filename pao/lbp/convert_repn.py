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
    #print(V.lower_bounds, V.upper_bounds)
    if V.upper_bounds is None:
        if V.lower_bounds is None:
            # variable unbounded
            for i in range(nxV):
                changes.append( VChangeUnbounded(real=i<nxR, v=i, w=nxV+i) )
            #print("HERE0")
            nxR = 2*nxR
            nxZ = 2*nxZ
        else:
            # variable bounded below
            for i in range(nxV):
                lb = V.lower_bounds[i]
                if lb == 0:
                    # Ignore non-negative variables
                    continue
                elif lb == np.NINF:
                    # bound is -infinity
                    if i<V.nxR:
                        changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                        nxR += 1
                    else:
                        changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                        nxZ += 1
                else:
                    # bound is constant
                    changes.append( VChangeLowerBound(real=i<V.nxR, v=i, lb=lb) )
    else:
        #print("YES")
        if V.lower_bounds is None:
            # Variables are unbounded below
            for i in range(nxV):
                ub = V.upper_bounds[i]
                if ub == np.PINF:
                    # Unbounded variable
                    if i<V.nxR:
                        changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                        nxR += 1
                    else:
                        changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                        nxZ += 1
                else:
                    changes.append( VChangeUpperBound(real=i<V.nxR, v=i, ub=ub) )
        else:
            #print("YES", nxR)
            # Variables are bounded
            for i in range(nxV):
                lb = V.lower_bounds[i]
                ub = V.upper_bounds[i]
                if ub == np.PINF:
                    if lb == 0:
                        continue
                    elif lb == np.NINF:
                        # Unbounded variable
                        if i<V.nxR:
                            changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                            #print("HERE1")
                            nxR += 1
                        else:
                            changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
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


def _process_changes(changes, V, c_, d, A, b, add_rows=False):
    # Copy c, inserting empty columns
    if c_ is not None:
        c = np.zeros(changes.nxR + changes.nxZ + V.nxB)
        for i in range(changes.nxR_old):
            c[i] = c_[i]
        for i in range(changes.nxZ_old):
            c[i + changes.nxR] = c_[i + changes.nxR_old]
        for i in range(V.nxB):
            c[i + changes.nxR + changes.nxZ] = c_[i + changes.nxR_old + changes.nxZ_old]
    else:
        c = None

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
        print(chg)
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
                #if Acsc.indptr.size == v+1:
                #    #print(chg)
                #    #print(v, chg.v, i, chg.w)
                #    #print(Acsc.shape)
                #    #print(Acsc.indptr)
                #    #print(Acsc.indices)
                #    #print(Acsc.data)
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    B[row, w] = - Acsc[row, v]
                    i += 1

    if nrows == 0:
        return c, d, None, b

    Bdok = dok_matrix((nrows, changes.nxR+changes.nxZ+V.nxB))
    # Collect the items from B
    for k,v in B.items():
        Bdok[k] = v
    # Merge in the items from A, shifting columns
    Adok = Acsc.todok()
    for k,v in Adok.items():
        i,j = k
        if j >= changes.nxR_old+changes.nxZ_old:
            j += (changes.nxR-changes.nxR_old) + (changes.nxZ-changes.nxZ_old)
        elif j >= changes.nxR_old:
            j += (changes.nxR-changes.nxR_old)
        Bdok[i,j] = v
    return c, d, Bdok.tocoo(), b


def convert_to_nonnegative_variables(ans, inequalities):
    U = ans.U
    L = U.L
    #
    # Collect real variables that are changing
    #
    UxV = U.x
    #print(UxV.nxR, UxV.nxZ, UxV.nxB, len(UxV))
    changes_U = _find_nonpositive_variables(UxV, inequalities)
    UxV.resize(changes_U.nxR, changes_U.nxZ, UxV.nxB)
    UxV.lower_bounds = np.zeros(changes_U.nxR + changes_U.nxZ + UxV.nxB)
    changes_L = {}
    for i in range(len(L)):
        LxV = L[i].x
        changes_L_ = _find_nonpositive_variables(LxV, inequalities)
        LxV.resize(changes_L_.nxR, changes_L_.nxZ, LxV.nxB)
        LxV.lower_bounds = np.zeros(changes_L_.nxR + changes_L_.nxZ + LxV.nxB)
        changes_L[i] = changes_L_
    #
    # Process changes related to upper-level variables
    #
    if len(changes_U) > 0:
        UxV = U.x
        #UcUxV = U.c.U.x #getattr(U.c.U, vstr)
        #UAUxV = U.A.U.x #getattr(U.A.U, vstr)
        U.c[U], U.d, U.A[U], U.b = \
                _process_changes(changes_U, UxV, U.c[U], U.d, U.A[U], U.b, add_rows=True)
        #setattr(U.c.U, vstr, UcUxV)
        #setattr(U.A.U, vstr, UAUxV)
        for i in range(len(L)):
            #LcUxV = getattr(L[i].c.U, vstr)
            #LAUxV = getattr(L[i].A.U, vstr)
            L[i].c[U], L[i].d, L[i].A[U], L[i].b = \
                _process_changes(changes_U, UxV, L[i].c[U], L[i].d, L[i].A[U], L[i].b)
            #setattr(L[i].c.U, vstr, LcUxV)
            #setattr(L[i].A.U, vstr, LAUxV)
    #
    # Process changes related to lower-level variables
    #
    for i in range(len(L)):
        LxV = L[i].x
        if len(changes_L[i]) > 0:
            #UcLxV = getattr(U.c.L[i], vstr)
            #UALxV = getattr(U.A.L[i], vstr)
            U.c[L[i]], U.d, U.A[L[i]], U.b = \
                    _process_changes(changes_L[i], LxV, U.c[L[i]], U.d, U.A[L[i]], U.b)
            #setattr(U.c.L[i], vstr, UcLxV)
            #setattr(U.A.L[i], vstr, UALxV)
            #
            #LcLxV = getattr(L[i].c.L[i], vstr)
            #LALxV = getattr(L[i].A.L[i], vstr)
            L[i].c[L[i]], L[i].d, L[i].A[L[i]], L[i].b = \
                    _process_changes(changes_L[i], LxV, L[i].c[L[i]], L[i].d, L[i].A[L[i]], L[i].b, add_rows=True)
            #setattr(L[i].c.L[i], vstr, LcLxV)
            #setattr(L[i].A.L[i], vstr, LALxV)
    #
    # Resize constraint matrices
    #
    # After processing upper and lower variables, we may have added constraints.  The other
    # upper/lower constraint matrices need to be resized as well.
    #
    for i in range(len(L)):
        if U.A[L[i]] is not None:
            U.A[L[i]].resize( [len(U.b), len(L[i].x)] )
        if L[i].A[U] is not None:
            L[i].A[U].resize( [len(L[i].b), len(U.x)] )
    #
    return changes_U, changes_L


def combine_matrices(A, B):
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


def convert_sense(level, U, L):
    level.minimize = True
    level.d *= -1
    if level.c[U] is not None:
        level.c[U] *= -1
    for i in range(len(L)):
        if level.c[L[i]] is not None:
            level.c[L[i]] *= -1


def convert_to_minimization(ans):
    if not ans.U.minimize:
        convert_sense(ans.U, ans.U, ans.U.L)
    for i in range(len(ans.U.L)):
        if not ans.U.L[i].minimize:
            convert_sense(ans.U.L[i], ans.U, ans.U.L)


def add_ineq_constraints(mat):
    if mat is None:
        return None
    x=mat.tocoo()
    nrows = mat.shape[0]
    d = -1 * x.data
    r = x.row #+ nrows
    c = x.col
    newmat = coo_matrix((d,(r,c)), shape=[nrows, mat.shape[1]])
    return vstack([mat, newmat])
    

def convert_constraints(ans, inequalities):
    U = ans.U
    L = U.L
    if inequalities:
        #
        # Creating inequality constraints from equalities by 
        # duplicating constraints
        #
        if not U.inequalities:
            bnew = np.copy(U.b)
            bnew *= -1
            U.b = np.concatenate((U.b, bnew))
            U.A[U] = add_ineq_constraints(U.A[U])
            for i in range(len(L)):
                U.A[L[i]] = add_ineq_constraints(U.A[L[i]])
        for i in range(len(L)):
            if not L[i].inequalities:
                bnew = np.copy(L[i].b)
                bnew *= -1
                L[i].b = np.concatenate((L[i].b, bnew))
                L[i].A[U] = add_ineq_constraints(L[i].A[U])
                L[i].A[L[i]] = add_ineq_constraints(L[i].A[L[i]])
    else:
        #
        # Add slack variables to create equality constraints from inequalities
        #
        if U.inequalities and len(U.b) > 0:
            nxR = U.x.nxR
            U.x.resize( nxR + len(U.b), U.x.nxZ, U.x.nxB, lb=0 )
            B = dok_matrix((len(U.b), len(U.x)))
            for k,v in U.A[U].todok().items():
                ii,jj = k
                if ii<nxR:
                    B[ii,jj] = v
                else:
                    B[ii,jj+len(U.b)] = v
            for i in range(len(U.b)):
                B[i,i+nxR] = 1
            for i in range(len(U.b)):
                if U.c[U] is not None:
                    U.c[U] = np.append(U.c[U], 0)
                for L_ in L:
                    if L_.c[U] is not None:
                        L_.c[U] = np.append(L_.c[U], 0)
            U.A[U] = B
            for i in range(len(L)):
                if L[i].A[U] is not None:
                    L[i].A[U].resize( (L[i].A[U].shape[0], U.x.nxR) )

        for i in range(len(L)):
            if L[i].inequalities and len(L[i].b) > 0:
                nxR = L[i].x.nxR
                L[i].x.resize( nxR + len(L[i].b), L[i].x.nxZ, L[i].x.nxB, lb=0 )
                B = dok_matrix((len(L[i].b), len(L[i].x)))
                #for k,v in L[i].A[U].todok().items():
                for k in range(len(L[i].b)):
                    B[k,j] = 1
                    j += 1
                    if U.c[L[i]] is not None:
                        U.c[L[i]] = np.append(U.c[L[i]], 0)
                    if L[i].c[L[i]] is not None:
                        L[i].c[L[i]] = np.append(L[i].c[L[i]], 0)
                L[i].A[L[i]] = combine_matrices(L[i].A[L[i]], B)

                if U.A[L[i]] is not None:
                    U.A[L[i]].resize( (U.A[L[i]].shape[0], L[i].x.nxR) )
    #
    # Update inequality values
    #
    U.inequalities = inequalities
    for i in range(len(L)):
        L[i].inequalities = inequalities


def get_multipliers(lbp, changes_U, changes_L):
    # 
    # If there were no changes, then the multiplier is 1
    #
    multipliers_U =   [[(i,1)] for i in lbp.U.x]
    multipliers_L = [ [[(i,1)] for i in lbp.U.L[j].x] for j in range(len(lbp.U.L)) ]
    for chg in changes_U:
        if type(chg) is VChangeUpperBound:
            multipliers_U[ chg.v ] = [(chg.v,-1)]
        elif type(chg) is VChangeUnbounded:
            multipliers_U[ chg.v ] = [(chg.v,1), (chg.w,-1)]
    for i in changes_L:
        for chg in changes_L[i]:
            if type(chg) is VChangeUpperBound:
                multipliers_L[i][ chg.v ] = [(chg.v,-1)]
            elif type(chg) is VChangeUnbounded:
                multipliers_L[i][ chg.v ] = [(chg.v,1), (chg.w,-1)]
    return multipliers_U, multipliers_L


def convert_binaries_to_integers(lbp):
    if len(lbp.U.xB) > 0:
        nxZ = len(lbp.U.xZ)
        nxB = len(lbp.U.xB)
        lbp.U.xZ.resize(nxZ+nxB, lb=0, ub=1)
        lbp.U.xB.resize(0)

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
    changes_Ux, changes_Lx = convert_to_nonnegative_variables(ans, inequalities)
    #
    # Setup multipliers that are used to convert variables back to the original model
    #
    multipliers_Ux, multipliers_Lx = get_multipliers(lbp, changes_Ux, changes_Lx)

    return ans, LBP_SolutionManager( multipliers_Ux, multipliers_Lx )

