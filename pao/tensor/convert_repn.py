from scipy.sparse import coo_matrix, dok_matrix
import numpy as np


def _find_nonpositive_variables(xR, inequalities):
    changes = []
    nxR = len(xR)
    nR = nxR
    if xR.upper_bounds is None:
        if xR.lower_bounds is None:
            # Variables are unbounded
            changes = [(i,4,nxR+i) for i in range(nxR)]
            nR = 2*nxR
        else:
            # Variables are unbounded above
            for i in range(nxR):
                lb = xR.lower_bounds[i]
                if lb == 0:
                    # Ignore non-negative variables
                    continue
                elif lb == np.NINF:
                    # Unbounded variable
                    changes.append( (i,4,nR) )
                    nR += 1
                else:
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
                else:
                    changes.append( (i,3,lb,ub,nR) )
                    nR += 1
    return changes, nR


def _process_changes(changes, c, A, b):
    c = copy.copy(lbp.U.c.U.xR)
    Acsc = lbp.U.A.U.xR.tocsc()
    b = ans.U.b
    nrows = A.size[0]
    A = {}
    for chg in changes_U:
        v = chg[0]
        if chg[1] == 1:     # real variable bounded below
            lb = chg[2]
            ans.U.d += c[v]*lb
        elif chg[1] == 2:   # real variable bounded above
            ub = chg[2]
            ans.U.d += c[v]*ub
            c[v] *= -1
        elif chg[1] == 3:   # real variable bounded
            lb = chg[2]
            ub = chg[3]
            w = chg[4]
            ans.U.d += c[v]*lb
            b.append(ub-lb)
            A[nrows, v] = 1
            nrows += 1
            if w is not None:
                c.append(0)
        else:               # real variable unbounded
            w = chg[2]
            i = Acsc.indptr[v]      # index of the vth column in the A matrix
            inext = Acsc.indptr[v+1]
            while i<inext:
                row = Acsc.indices[i]
                A[w, row] = - Acsc[v, row]
            c.append(0)
    return c, A.tocoo(), b


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
    U.minimize = lbp.U.minimize
    U.d = lbp.U.d
    ans.U.b = copy.copy(lbp.U.b)
    for i,L in enumerate(lbp.L):
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
    changes_U, nR = _find_nonpositive_variables(lbp.U.xR, lbp.U.inequalities)
    nR += lbp.U.inequalities*len(lbp.U.b)
    U = ans.add_upper(nxR=nR, nxZ=len(lbp.U.xZ), nxB=len(lbp.U.xB))
    U.xR.lower_bounds = np.zeros(nR)
    changes_L = []
    for L in lbp.L:
        changes_L_, nR = _find_nonpositive_variables(L.xR, L.inequalities)
        nR += L.inequalities*len(L.b)
        LL = ans.add_lower(nxR=nR, nxZ=len(L.xZ), nxB=len(L.xB))
        LL.xR.lower_bounds = np.zeros(nR)
        changes_L.append(changes_L_)
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
        ans.U.c.U.xR, ans.U.A.U.xR, ans.U.b = \
                _process_changes(changes, lbp.U.c.U.xR, lbp.U.A.U.xR, lbp.U.b)
        for i,L in enumerate(lbp.L):
            ans.L[i].c.U.xR, ans.L[i].A.U.xR, ans.L[i].b = \
                _process_changes(changes, L.c.U.xR, L.A.U.xR, L.b)
    #
    # Process changes related to lower-level variables
    #
    for i,L in enumerate(lbp.L):
        if len(changes_L[i]) == 0:
            ans.U.c.L.xR = copy.copy(lbp.U.c.L.xR)
            ans.U.A.L.xR = copy.copy(lbp.U.A.L.xR)
            for i,L in enumerate(lbp.L):
                ans.L[i].c.L.xR = copy.copy(L.c.L.xR)
                ans.L[i].A.L.xR = copy.copy(L.A.L.xR)
        else:
            ans.U.c.L.xR, ans.U.A.L.xR, ans.L.b = \
                    _process_changes(changes, lbp.U.c.L.xR, lbp.U.A.L.xR, lbp.L.b)
            for i,L in enumerate(lbp.L):
                ans.L[i].c.L.xR, ans.L[i].A.L.xR, ans.L[i].b = \
                    _process_changes(changes, L.c.L.xR, L.A.L.xR, L.b)
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
        ans.U.A.U.xR = ans.U.A.U.xR + B
    ans.U.inequalities = False
    for i,L in enumerate(lbp.L):
        if L.inequalities:
            B = dok_matrix((len(ans.L.b), len(ans.L.xR)))
            j = len(ans.L.xR)-len(ans.L.b)
            for i in range(len(ans.L.b)):
                B[i,j] = 1
            ans.L.A.L.xR = ans.L.A.L.xR + B
        ans.L[i].inequalities = False

