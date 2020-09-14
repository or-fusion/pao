import copy
from scipy.sparse import coo_matrix, dok_matrix, csc_matrix
import numpy as np
from .repn import LinearBilevelProblem


def _find_nonpositive_variables(xR, inequalities):
    changes = []
    nxR = len(xR)
    nRbounded=0     # no. variable added, which are slacks added later
    nR = nxR
    if xR.upper_bounds is None:
        if xR.lower_bounds is None:
            # real variable unbounded
            changes = [(i,4,nxR+i) for i in range(nxR)]
            nR = 2*nxR
        else:
            # real variable bounded below
            for i in range(nxR):
                lb = xR.lower_bounds[i]
                if lb == 0:
                    # Ignore non-negative variables
                    continue
                elif lb == np.NINF:
                    # bound is -infinity
                    changes.append( (i,4,nR) )
                    nR += 1
                else:
                    # bound is constant
                    changes.append( (i,1,lb) )
    else:
        if xR.lower_bounds is None:
            # Variables are unbounded below
            for i in range(nxR):
                ub = xR.upper_bounds[i]
                if ub == np.PINF:
                    # Unbounded variable
                    changes.append( (i,4,nR) )
                    nR += 1
                else:
                    changes.append( (i,2,ub) )
        else:
            # Variables are bounded
            for i in range(nxR):
                lb = xR.lower_bounds[i]
                ub = xR.upper_bounds[i]
                if ub == np.PINF:
                    if lb == 0:
                        continue
                    elif lb == np.NINF:
                        # Unbounded variable
                        changes.append( (i,4,nR) )
                        nR += 1
                    else:
                        changes.append( (i,1,lb) )
                elif lb == np.NINF:
                    changes.append( (i,2,ub) )
                elif inequalities:
                    # Can create an inequality constraint
                    changes.append( (i,3,lb,ub,None) )
                    nRbounded += 1     # slack variable
                else:
                    changes.append( (i,3,lb,ub,nR) )
                    nR += 1
    return changes, nR+nRbounded


def _process_changes(changes, nxR, c, d, A, b, level_vars=True):
    #assert (A is not None), "Only process changes when we have constraints"
    #if A is None:
    #    c = copy.copy(c)
    #    d = copy.copy(d)
    #    b = copy.copy(b)
    #    return c, d, A, b

    c = copy.copy(c)
    d = copy.copy(d)
    #if b is None:
    #    b = np.ndarray(0)
    #else:
    b = copy.copy(b)

    if A is None:
        Acsc = csc_matrix(0)
        nrows = 0
    else:
        Acsc = A.tocsc()
        nrows = A.shape[0]

    B = {}
    for chg in changes:
        v = chg[0]
        if chg[1] == 1:     # real variable bounded below
            lb = chg[2]
            if c is not None:
                d += c[v]*lb
            if A is not None:
                i = Acsc.indptr[v]      # index of the vth column in the A matrix
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    b[row] -= Acsc[row, v]*lb
                    i += 1
        elif chg[1] == 2:   # real variable bounded above
            ub = chg[2]
            if c is not None:
                d += c[v]*ub
                c[v] *= -1
            if A is not None:
                i = Acsc.indptr[v]      # index of the vth column in the A matrix
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    b[row] -= Acsc[row, v]*ub
                    Acsc[row, v] *= -1
                    i += 1
        elif chg[1] == 3:   # real variable bounded
            lb = chg[2]
            ub = chg[3]
            w = chg[4]
            if c is not None:
                d += c[v]*lb
                if w is not None:
                    c = np.append(c, 0)
            if A is not None:
                i = Acsc.indptr[v]      # index of the vth column in the A matrix
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    b[row] -= Acsc[row, v]*lb
                    i += 1
            if level_vars:
                # Add new constraint
                # If w is not None, then we are adding an associated slack variable
                # NOTE: We only add the constraint to the level that "owns" the variables
                b = np.append(b, ub-lb)
                B[nrows, v] = 1
                if w is not None:
                    B[nrows, w] = 1
                nrows += 1
        else:               # real variable unbounded
            w = chg[2]
            if c is not None:
                c = np.append(c, -c[v])
            if A is not None:
                i = Acsc.indptr[v]      # index of the vth column in the A matrix
                inext = Acsc.indptr[v+1]
                while i<inext:
                    row = Acsc.indices[i]
                    B[row, w] = - Acsc[row, v]
                    i += 1
    if nrows == 0:
        return c, d, None, b
    Bdok = dok_matrix((nrows, nxR))
    for k,v in B.items():
        Bdok[k] = v
    A = combine_matrices(Acsc, Bdok)
    return c, d, A.tocoo(), b


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
    #print("A")
    #print(A)
    #print("B")
    #print(B)
    x=A.tocoo()
    y=B.tocoo()
    d = np.concatenate((x.data, y.data))
    r = np.concatenate((x.row, y.row))
    c = np.concatenate((x.col, y.col))
    ans = coo_matrix((d,(r,c)), shape=shape)
    #print("A + B")
    #print(ans)
    return ans


def convert_LinearBilevelProblem_to_standard_form(lbp):
    """
    After applying this transformation, the problem has the form:
        1. Each real variable x is nonnegative (x >= 0)
        2. Constraints are equalities
    Thus, if a level only has real variables, it will be in standard form
    following this transformation.
    """
    #
    # Setup converted problem
    #
    ans = LinearBilevelProblem(name=lbp.name)
    ans.add_upper(nxZ=len(lbp.U.xZ), nxB=len(lbp.U.xB))
    ans.U.inequalities = False
    ans.U.minimize = lbp.U.minimize
    ans.U.d = lbp.U.d
    ans.U.b = copy.copy(lbp.U.b)
    for i,L in enumerate(lbp.L):
        ans.add_lower(nxZ=len(L.xZ), nxB=len(L.xB))
        ans.L[i].inequalities = False
        ans.L[i].d = L.d
        ans.L[i].b = copy.copy(L.b)
    #
    # Copy the data associated with integer and binary variables
    #
    ans.U.xZ.lower_bounds = copy.copy(lbp.U.xZ.lower_bounds)
    ans.U.xZ.upper_bounds = copy.copy(lbp.U.xZ.upper_bounds)
    ans.U.c.U.xZ = copy.copy(lbp.U.c.U.xZ)
    ans.U.c.U.xB = copy.copy(lbp.U.c.U.xB)
    ans.U.A.U.xZ = copy.copy(lbp.U.A.U.xZ)
    ans.U.A.U.xB = copy.copy(lbp.U.A.U.xB)
    for i,L in enumerate(lbp.L):
        ans.L[i].xZ.lower_bounds = copy.copy(L.xZ.lower_bounds)
        ans.L[i].xZ.upper_bounds = copy.copy(L.xZ.upper_bounds)
        ans.L[i].c.U.xZ = copy.copy(L.c.U.xZ)
        ans.L[i].c.U.xB = copy.copy(L.c.U.xB)
        ans.L[i].A.U.xZ = copy.copy(L.A.U.xZ)
        ans.L[i].A.U.xB = copy.copy(L.A.U.xB)
    #
    # Collect real variables that are changing
    #
    if lbp.U.A.U.xR is not None and lbp.U.A.U.xR.shape[0] > 0:
        changes_U, nR = _find_nonpositive_variables(lbp.U.xR, lbp.U.inequalities)
        nR += lbp.U.inequalities*len(lbp.U.b)
        ans.U.xR.resize(nR)
        ans.U.xR.lower_bounds = np.zeros(nR)
    else:
        changes_U = []
        ans.U.xR.resize(len(lbp.U.xR))
    changes_L = {}
    for i in range(len(lbp.L)):
        if lbp.L[i].A.L[i].xR is not None and lbp.L[i].A.L[i].xR.shape[0] > 0:
            changes_L_, nR = _find_nonpositive_variables(lbp.L[i].xR, lbp.L[i].inequalities)
            nR += lbp.L[i].inequalities*len(lbp.L[i].b)
            ans.L[i].xR.resize(nR)
            ans.L[i].xR.lower_bounds = np.zeros(nR)
            changes_L[i] = changes_L_
        else:
            ans.L[i].xR.resize(len(L.xR))
    #
    # Process changes related to upper-level variables
    #
    if len(changes_U) == 0:
        ans.U.c.U.xR = copy.copy(lbp.U.c.U.xR)
        ans.U.A.U.xR = copy.copy(lbp.U.A.U.xR)
        for i,L in enumerate(lbp.L):
            ans.L[i].c.U.xR = copy.copy(L.c.U.xR)
            ans.L[i].A.U.xR = copy.copy(L.A.U.xR)
    else:
        ans.U.c.U.xR, ans.U.d, ans.U.A.U.xR, ans.U.b = \
                _process_changes(changes_U, len(ans.U.xR), lbp.U.c.U.xR, lbp.U.d, lbp.U.A.U.xR, lbp.U.b)
        for i,L in enumerate(lbp.L):
            ans.L[i].c.U.xR, ans.L[i].d, ans.L[i].A.U.xR, ans.L[i].b = \
                _process_changes(changes_U, len(ans.U.xR), L.c.U.xR, L.d, L.A.U.xR, L.b, level_vars=False)
    #
    # Process changes related to lower-level variables
    #
    for i,L in enumerate(lbp.L):
        if not i in changes_L:
            continue
        if len(changes_L[i]) == 0:
            ans.U.c.L[i].xR = copy.copy(lbp.U.c.L[i].xR)
            ans.U.A.L[i].xR = copy.copy(lbp.U.A.L[i].xR)
            ans.L[i].c.L[i].xR = copy.copy(lbp.L[i].c.L[i].xR)
            ans.L[i].A.L[i].xR = copy.copy(lbp.L[i].A.L[i].xR)
        else:
            ans.U.c.L[i].xR, ans.U.d, ans.U.A.L[i].xR, ans.U.b = \
                    _process_changes(changes_L[i], len(ans.L[i].xR), lbp.U.c.L[i].xR, ans.U.d, lbp.U.A.L[i].xR, ans.U.b, level_vars=False)
            ans.L[i].c.L[i].xR, ans.L[i].d, ans.L[i].A.L[i].xR, ans.L[i].b = \
                    _process_changes(changes_L[i], len(ans.L[i].xR), L.c.L[i].xR, ans.L[i].d, L.A.L[i].xR, ans.L[i].b)
    #
    # Resize constraint matrices
    #
    # After processing upper and lower variables, we may have added constraints.  The other
    # upper/lower constraint matrices need to be resized as well.
    #
    for i in range(len(ans.L)):
        if ans.U.A.L[i].xR is not None:
            ans.U.A.L[i].xR.resize( [len(ans.U.b), len(ans.L[i].xR)] )
        if ans.L[i].A.U.xR is not None:
            ans.L[i].A.U.xR.resize( [len(ans.L[i].b), len(ans.U.xR)] )
    #
    # Add slack variables if the constraints are defined with inequalities
    #
    # Note that we already added the variable above, so we just need to add the 
    # nonzeros to the constraint matrix.
    #
    if lbp.U.inequalities:
        B = dok_matrix((len(ans.U.b), len(ans.U.xR)))
        j = len(ans.U.xR)-len(ans.U.b)
        for i in range(len(ans.U.b)):
            B[i,j] = 1
            j += 1
            if ans.U.c.U.xR is not None:
                ans.U.c.U.xR = np.append(ans.U.c.U.xR, 0)
            for k,L in enumerate(ans.L):
                if ans.L[k].c.U.xR is not None:
                    ans.L[k].c.U.xR = np.append(ans.L[k].c.U.xR, 0)
        ans.U.A.U.xR = combine_matrices(ans.U.A.U.xR, B)

    for i in range(len(lbp.L)):
        if lbp.L[i].inequalities:
            B = dok_matrix((len(ans.L[i].b), len(ans.L[i].xR)))
            j = len(ans.L[i].xR)-len(ans.L[i].b)
            for k in range(len(ans.L[i].b)):
                B[k,j] = 1
                j += 1
                if ans.U.c.L[i].xR is not None:
                    ans.U.c.L[i].xR = np.append(ans.U.c.L[i].xR, 0)
                if ans.L[i].c.L[i].xR is not None:
                    ans.L[i].c.L[i].xR = np.append(ans.L[i].c.L[i].xR, 0)
            ans.L[i].A.L[i].xR = combine_matrices(ans.L[i].A.L[i].xR, B)
    #
    return ans

