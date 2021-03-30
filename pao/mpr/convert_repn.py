import copy
from scipy.sparse import coo_matrix, dok_matrix, csc_matrix, vstack
import numpy as np
from .repn import LinearMultilevelProblem, QuadraticMultilevelProblem, LinearLevelRepn
from .soln_manager import LMP_SolutionManager, SolutionManager_Linearized_Bilinear_Terms

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

    def __init__(self, nxR, nxZ):
        self._data = []     # List of VChange object
        self.nxR_old = nxR  # The old nxR value before applying changes
        self.nxZ_old = nxZ  # The old nxZ value before applying changes
        self.nxR = nxR      # The new nxR value after applying changes
        self.nxZ = nxZ      # The new nxZ value after applying changes

    def append(self, chg):
        self._data.append(chg)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for chg in self._data:
            yield chg

def _matrix_rows(M, col):
    if M is None or M.data.size == 0:
        return
    i = M.indptr[col]
    inext = M.indptr[col+1]
    while i<inext:
        row = M.indices[i]
        yield row
        i += 1

def _find_nonpositive_variables(V, inequalities):
    nxV = V.nxR+V.nxZ
    nxR = V.nxR
    nxZ = V.nxZ
    changes = VChanges(nxR, nxZ)

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
                    nxR += 1
                else:
                    changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                    nxZ += 1
            else:
                # Bounded below
                changes.append( VChangeLowerBound(real=i<V.nxR, v=i, lb=lb) )
        elif lb == np.NINF:
            # Bounded above
            changes.append( VChangeUpperBound(real=i<V.nxR, v=i, ub=ub) )
        elif inequalities:
            # Bounded above and below (inequality formulation)
            changes.append( VChangeRange(real=i<V.nxR, v=i, lb=lb, ub=ub) )
        else:
            # Bounded above and below (equality formulation)
            changes.append( VChangeRange(real=i<V.nxR, v=i, lb=lb, ub=ub, w=nxR) )
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
    return changes


def _process_changes_obj(changes, V, c, d):
    if c is None:
        return c, d

    d = copy.copy(d)

    for chg in changes:
        v = chg.v
        if type(chg) is VChangeLowerBound:      # real variable bounded below
            # Replace v >= lb with v' >= 0
            # v = lb + v'
            # c[v]*v = c[v]*lb + c[v]*v'
            lb = chg.lb
            d += c[v]*lb

        elif type(chg) is VChangeUpperBound:    # real variable bounded above
            # Replace v <= ub with v' >= 0
            # v = ub - v'
            # c[v]*v = c[v]*ub - c[v]*v'
            ub = chg.ub
            d += c[v]*ub
            c[v] *= -1

        elif type(chg) is VChangeRange:         # real variable bounded
            # Replace lb <= v <= ub with v' >= 0
            # v = lb + v'
            # c[v]*v = c[v]*lb + c[v]*v'
            lb = chg.lb
            ub = chg.ub
            d += c[v]*lb
            w = chg.w
            if w is not None:
                c[w] = 0

        else:                                   # real variable unbounded
            # Replace unbounded v with v',v'' >= 0
            # v = v' - v''
            # c[v]*v = c[v]*v' - c[v]*v''
            w = chg.w
            c[w] = -c[v]

    return c, d


def _process_changes_con(changes, V, A, b, add_rows=False):
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
            # Replace v >= lb with v' >= 0
            # v = lb + v'
            # A[row,v]*v = A[row,v]*lb + A[row,v]*v'
            lb = chg.lb
            for row in _matrix_rows(Acsc, v):
                b[row] -= Acsc[row, v]*lb

        elif type(chg) is VChangeUpperBound:    # real variable bounded above
            # Replace v <= ub with v' >= 0
            # v = ub - v'
            # A[row,v]*v = A[row,v]*ub - A[row,v]*v'
            ub = chg.ub
            for row in _matrix_rows(Acsc, v):
                b[row] -= Acsc[row, v]*ub
                Acsc[row, v] *= -1

        elif type(chg) is VChangeRange:         # real variable bounded
            # Replace lb <= v <= ub with v' >= 0
            # v = lb + v'
            # A[row,v]*v = A[row,v]*lb + A[row,v]*v'
            lb = chg.lb
            ub = chg.ub
            w = chg.w
            for row in _matrix_rows(Acsc, v):
                b[row] -= Acsc[row, v]*lb
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
            # Replace unbounded v with v',v'' >= 0
            # v = v' - v''
            # A[row,v]*v = A[row,v]*v' - A[row,v]*v''
            w = chg.w
            for row in _matrix_rows(Acsc, v):
                B[row, w] = - Acsc[row, v]

    if nrows == 0:
        return None, b

    Bdok = dok_matrix((nrows, changes.nxR+changes.nxZ+V.nxB))
    # Collect the items from B
    for k,v in B.items():
        Bdok[k] = v
    # Merge in the items from A, shifting columns
    Adok = Acsc.todok()
    for k,v in Adok.items():
        Bdok[k] = v
    return Bdok.tocoo(), b


def X_process_changes_P(changes, Lx, Xci, P, Xcj, changes_i, changes_j): #pragma: nocover
    if P is None:
        return Xci, P, Xcj

    if Xci is None:
        Xci = np.zeros(P.shape[0])
    if Xcj is None:
        Xcj = np.zeros(P.shape[1])

    if changes_i and changes_j:
        raise RuntimeError("PAO does not (yet) support quadratic and bilinear terms amongst variables in the same level")

    elif changes_j:
        Pcsc = P.tocsc()

        B = {}
        for chg in changes:
            v = chg.v
            if type(chg) is VChangeLowerBound:      # real variable bounded below
                # Replace v >= lb with v' >= 0
                # v = lb + v'
                # P[row,v]*v = P[row,v]*lb + P[row,v]*v'
                lb = chg.lb
                for row in _matrix_rows(Pcsc, v):
                    Xci[row] += Pcsc[row, v]*lb

            elif type(chg) is VChangeUpperBound:    # real variable bounded above
                # Replace v <= ub with v' >= 0
                # v = ub - v'
                # P[row,v]*v = P[row,v]*ub - P[row,v]*v'
                ub = chg.ub
                for row in _matrix_rows(Pcsc, v):
                    Xci[row] += pcsc[row, v]*ub
                    Pcsc[row, v] *= -1

            elif type(chg) is VChangeRange:         # real variable bounded
                # Replace lb <= v <= ub with v' >= 0
                # v = lb + v'
                # P[row,v]*v = P[row,v]*lb + P[row,v]*v'
                lb = chg.lb
                for row in _matrix_rows(Pcsc, v):
                    Xci[row] += Pcsc[row, v]*lb

            else:                                   # real variable unbounded
                # Replace unbounded v with v',v'' >= 0
                # v = v' - v''
                # A[row,v]*v = A[row,v]*v' - A[row,v]*v''
                w = chg.w
                for row in _matrix_rows(Pcsc, v):
                    B[row, w] = - Pcsc[row, v]

        Bdok = dok_matrix((Pcsc.shape[0], changes.nxR+changes.nxZ+Lx.nxB))
        # Collect the items from B
        for k,v in B.items():
            Bdok[k] = v
        # Merge in the items from Pcsc
        Pdok = Pcsc.todok()
        for k,v in Pdok.items():
            Bdok[k] = v

        if len(Xci.nonzero()) == 0:
            Xci = None
        if len(Xcj.nonzero()) == 0:
            Xcj = None

        return Xci, Bdok.tocoo(), Xcj

    else:   # changes_i
        #print("HERE")
        _Xcj, _P, _Xci = _process_changes_P(changes, Lx, Xcj, P.transpose(), Xci, changes_j, changes_i)
        #print(Xci, _Xci)
        #print(Xcj, _Xcj)
        return _Xci, _P.transpose(), _Xcj

def X_process_changes(changes, V, c, d, A, b, add_rows=False):  #pragma: nocover
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
            # Replace v >= lb with v' >= 0
            # v' = v - lb
            lb = chg.lb
            if c is not None:
                # c[v]*v = c[v]*lb + c[v]*v'
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
    # Iterate over all levels in the model.  For each level,
    # collect the changes needed to make the variables non-negative.
    #
    changes = {}
    for L in ans.levels():
        changes[L.id] = _find_nonpositive_variables(L.x, inequalities)
    #
    # Process changes 
    #
    # Iterate over all levels of the model.  For each level,
    # resize the variables and set the lower bounds.  Then iterate over the levels that
    # could reference those variables, and update the data structures in those
    # levels.
    #
    for L in ans.levels():
        L.resize(nxR=changes[L.id].nxR, nxZ=changes[L.id].nxZ, nxB=L.x.nxB)
        L.x.lower_bounds = np.zeros(len(L.x))
        L.x.upper_bounds = [np.PINF]*(changes[L.id].nxR+changes[L.id].nxZ) + [1]*L.x.nxB
        if len(changes[L.id]) > 0:
            for X in L.levels():
                X.c[L], X.d = _process_changes_obj(changes[L.id], L.x, X.c[L], X.d)
                X.A[L], X.b = _process_changes_con(changes[L.id], L.x, X.A[L], X.b, add_rows=L.id == X.id)
                #
                # NOTE: Conversion of a quadratic multilevel problem is not supported
                #
                #if quadratic:
                #    for i,j in X.P:
                #        X.c[i], X.P[i,j], X.c[j] = _process_changes_P(changes[L.id], L.x, X.c[i], X.P[i,j], X.c[j], i == L.id, j==L.id)
    return changes


def Xcombine_matrices(A, B):         #pragma: nocover
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
        #if type(L) is QuadraticLevelRepn:
        #    for i,j in L.P:
        #        L.P[i,j] = L.P[i,j].multiply(-1)


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


def get_multipliers(mpr, changes):
    multipliers = {}
    for L in mpr.levels():
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

def get_offsets(mpr, changes):
    offsets = {}
    for L in mpr.levels():
        #
        # If there were no changes, then the offset is 0
        #
        offsets[L.id] =   [0 for i in L.x]
        for chg in changes[L.id]:
            if type(chg) is VChangeUpperBound:
                offsets[L.id][ chg.v ] = chg.ub
            elif type(chg) is VChangeLowerBound:
                offsets[L.id][ chg.v ] = chg.lb
            elif type(chg) is VChangeRange:
                offsets[L.id][ chg.v ] = chg.lb
    return offsets


def convert_binaries_to_integers(mpr, nonnegative=True):
    for L in mpr.levels():
        if L.x.nxB > 0:
            if nonnegative:
                nxRZ = L.x.nxR + L.x.nxZ
                nxB = L.x.nxB
                L.x._resize(nxR=L.x.nxR, nxZ=L.x.nxZ+L.x.nxB, nxB=0, lb=0, ub=np.PINF)
                if L.b is None or L.b.size == 0:
                    continue
                M = dok_matrix((nxB,len(L.x)))
                for i in range(nxB):
                    M[i,nxRZ+i] = 1
                L.b = np.append(L.b, [1]*nxB)
                if L.A[L] is None:
                    L.A[L] = M.tocoo()
                else:
                    L.A[L] = vstack([L.A[L], M.tocoo()])
            else:
                L.x._resize(nxR=L.x.nxR, nxZ=L.x.nxZ+L.x.nxB, nxB=0, lb=0, ub=1)
    #
    # Resize Matrices
    #
    for L in mpr.levels():
        for X in L.levels():
            A = X.A[L]
            if A is not None:
                A.resize( [len(X.b), len(L.x)] )


def convert_to_standard_form(M, inequalities=False):
    """
    Normalize the LinearMultilevelProblem into a standard form.

    This function copies the multilevel problem, **M**, and transforms
    the problem such that

    1. Each real variable x is nonnegative (x >= 0)
    2. Constraints have the specified form (e.g. all equalities or all inequalities)
    3. Each level is a minimization problem

    Args
    ----
    M : LinearMultilevelProblem
        The model that is being normalized
    inequalities : bool, Default: False
        If this is True, then the normalized form has inequality constraints.  Otherwise, the normalized
        form has equality constraints.

    Returns
    -------
    LinearMultilevelProblem
        A normalized version of the input model
    """
    # TODO - Linearize?  Or check that the problem is linear?
    #assert (type(M) is LinearMultilevelProblem), "Expected linear multilevel problem"

    #
    # Clone the object
    #
    ans = M.clone()
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

    return ans, LMP_SolutionManager(get_multipliers(M, changes), get_offsets(M, changes))


def merge_matrices(M1, M2, nrows, ncols):
    if M1 is None:
        M = dok_matrix((0,0))
    else:
        M = M1.todok()
    M.resize( (max(M.shape[0], nrows), max(M.shape[1], ncols)) )
    for k,v in M2.items():
        M[k] = v
    if M.shape[0] == 0 or M.shape[1] == 0:
        return None
    return M 


def linearize_bilinear_terms(M, bigM):
    """
    Generate a linear multilevel problem from a quadratic multilevel
    problem by linearizing bilinear terms.

    This function copies the linear terms in the multilevel problem,
    **M**, and replaces bilinear terms with a new variable.  This
    transformation only applies when at least one of the variables in
    each bilinear term is binary or integer.

    Args
    ----
    M : QuadraticMultilevelProblem
        The model that is being linearized
    bigM : float
        The big-M value used to linearize nonlinear terms

    Returns
    -------
    LinearMultilevelProblem
        A linearized version of the input model
    """
    assert (type(M) is QuadraticMultilevelProblem), "Expected quadratic multilevel problem"
    for L in M.levels():
        assert (L.inequalities), "The function linearize_bilinear_terms can only handle QMPs with inequalities"

    #
    # Explicit clone logic, since we are converting a Quadratic to a Linear representation
    #
    #ans = M.clone(clone_fn=_clone_level)
    ans = LinearMultilevelProblem()
    ans.name = M.name
    ans.U = M.U.clone(clone_fn=LinearLevelRepn._clone_level)
    ans.check()

    #
    # Collect all of the levels in the model
    #
    LL = {L.id:L for L in ans.levels()}
    #
    # Collect all of the bilevel terms
    #
    # The terms in P[i,j] and Q[i,j] generate a variable in level j,
    # regardless where they appear in the model.  Hence, we need to collect
    # these terms before adding their replacement throughout the model.
    #
    bilevel = {} 
    for L in M.levels():
        bilevel[L.id] = {}
    for L in M.levels():
        l = L.id
        for i,j in L.P:
            for v1,v2 in L.P[i,j].keys():
                assert (v1 >= LL[i].x.nxR+LL[i].x.nxZ), "Expected binary variable %d in bilinear term %s.P[%d,%d]" % (v1,str(L),i,j)
                if (i,v1,j,v2) not in bilevel[j]:
                    bilevel[j][i,v1,j,v2] = len(bilevel[j])
        for i,j in L.Q:
            for c,Q in enumerate(L.Q[i,j]):
                if Q is None:
                    continue
                for v1,v2 in Q.todok().keys():
                    assert (v1 >= LL[i].x.nxR+LL[i].x.nxZ), "Expected binary variable %d in bilinear term %s.Q[%d,%d][%d,%d]" % (v1,L.name,i,j,v1,v2)
                    if (i,v1,j,v2) not in bilevel[j]:
                        #print(i,j,c,v1,v2, L.Q[i,j][c][v1,v2])
                        bilevel[j][i,v1,j,v2] = len(bilevel[j])
    #
    # Return if no bilevel terms were found
    #
    if sum(len(bilevel[i]) for i in bilevel) == 0:
        return ans
    #
    # Now we walk through each level
    #
    # Add constraint terms for the existing variables that are in each term
    #
    # NOTE: We cache the constraint terms for the *new* variables
    #
    A = {}
    wlb = {}
    wub = {}
    for l,L in LL.items():
        lenb = len(L.b)
        nxR = L.x.nxR
        nrows = lenb
        wlb[l] = {}
        wub[l] = {}
        b = []
        # A[l] is the new terms in the constraint matrix for new variables in level l
        A[l] = {}
        # B[i] is the new terms in the constraint matrix for variables in level l
        X = {}
        # C[j] is the new terms in the constraint matrix for variables in level j that appear in level l
        Y = {}
        for key,w in bilevel[l].items():
            # w = xy
            i,v1,j,v2 = key
            assert (j==l), "Something is wrong..."
            if i not in X:
                X[i] = {}
            if j not in Y:
                Y[j] = {}
            lb = LL[j].x.lower_bounds[v2]
            wlb[j][nxR+w] = lb
            if lb == np.NINF:
                lb = -bigM
            ub = LL[j].x.upper_bounds[v2]
            wub[j][nxR+w] = ub
            if ub == np.PINF:
                ub = bigM
            # Lx - w <= 0
            X[i][nrows,v1] = lb
            A[l][nrows,nxR+w] = -1
            b.append(0)
            nrows += 1
            # Ux + y - w <= U
            Y[j][nrows,v2] = 1
            X[i][nrows,v1] = ub
            A[l][nrows,nxR+w] = -1
            b.append(ub)
            nrows += 1
            # w - Ux <= 0
            X[i][nrows,v1] = -ub
            A[l][nrows,nxR+w] = 1
            b.append(0)
            nrows += 1
            # w - y - Lx <= -L
            Y[j][nrows,v2] = -1
            X[i][nrows,v1] = -lb
            A[l][nrows,nxR+w] = 1
            if lb == 0:
                b.append(0)
            else:
                b.append(-lb)
            nrows += 1
        L.b = list(L.b) + b
        for i in X:
            L.A[i] = merge_matrices(L.A[i], X[i], len(L.b), len(LL[i].x))
        for j in Y:
            L.A[j] = merge_matrices(L.A[j], Y[j], len(L.b), len(LL[j].x))
    #
    # Resize the variables
    #
    for l,L in LL.items():
        L.resize(nxR=L.x.nxR+len(bilevel[l]), nxZ=L.x.nxZ, nxB=L.x.nxB)
        #for i in wlb[l]:
        #    L.x.lower_bounds[i] = wlb[l][i]
        #    L.x.upper_bounds[i] = wub[l][i]
    #
    # Update the coefficients of the objectives
    #
    nxR = {L.id:L.x.nxR for L in M.levels()}
    for L in M.levels():
        l = L.id
        for i,j in L.P:
            for v1,v2 in L.P[i,j].keys():
                w = bilevel[j][i,v1,j,v2]
                # The coefficient in ans at level l for variables in level j at (w + number of reals in M) is coef
                LL[l].c[j][w+nxR[j]] = L.P[i,j][v1,v2]
    #
    # Merge the cached terms now that we've shifted the variables
    #
    for l,L in LL.items():
        L.A[l] = merge_matrices(L.A[l], A[l], len(L.b), len(L.x))
    #
    # Update the A matrices with coefficients from Q[i,j]
    #
    for L in M.levels():
        l = L.id
        for i,j in L.Q:
            A = {}
            for c,Q in enumerate(L.Q[i,j]):
                if Q is None:
                    continue
                for v1,v2 in Q.todok().keys():
                    w = bilevel[j][i,v1,j,v2]
                    A[c,w+nxR[j]] = Q[v1,v2]
            LL[l].A[j] = merge_matrices(LL[l].A[j], A, len(LL[l].b), len(LL[j].x))

    return ans, SolutionManager_Linearized_Bilinear_Terms()

