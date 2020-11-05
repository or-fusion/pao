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


def _find_nonpositive_variables(V, inequalities):
    changes = []
    nxV = V.nxR+V.nxZ
    nxR = V.nxR
    nxZ = V.nxZ
    if V.upper_bounds is None:
        if V.lower_bounds is None:
            # variable unbounded
            changes = [VChangeUnbounded(real=i<nxR, v=i, w=nxV+i) for i in range(nxV)]
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
                    if i<nxR:
                        changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                        nxR += 1
                    else:
                        changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                        nxZ += 1
                else:
                    # bound is constant
                    changes.append( VChangeLowerBound(real=i<nxR, v=i, lb=lb) )
    else:
        if V.lower_bounds is None:
            # Variables are unbounded below
            for i in range(nxV):
                ub = V.upper_bounds[i]
                if ub == np.PINF:
                    # Unbounded variable
                    if i<nxR:
                        changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                        nxR += 1
                    else:
                        changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                        nxZ += 1
                else:
                    changes.append( VChangeUpperBound(real=i<nxR, v=i, ub=ub) )
        else:
            # Variables are bounded
            for i in range(nxV):
                lb = V.lower_bounds[i]
                ub = V.upper_bounds[i]
                if ub == np.PINF:
                    if lb == 0:
                        continue
                    elif lb == np.NINF:
                        # Unbounded variable
                        if i<nxR:
                            changes.append( VChangeUnbounded(real=True, v=i, w=nxR) )
                            nxR += 1
                        else:
                            changes.append( VChangeUnbounded(real=False, v=i, w=nxZ) )
                            nxZ += 1
                    else:
                        changes.append( VChangeLowerBound(real=i<nxR, v=i, lb=lb) )
                elif lb == np.NINF:
                    changes.append( VChangeUpperBound(real=i<nxR, v=i, ub=ub) )
                elif inequalities:
                    changes.append( VChangeRange(real=i<nxR, v=i, lb=lb, ub=ub) )
                else:
                    changes.append( VChangeRange(real=i<nxR, v=i, lb=lb, ub=ub, w=nxR) )
                    nxR += 1

    # Reset the variable id for integers, given the final value of nxR
    for c in changes:
        if type(c) is VChangeUnbounded and c.real is False:
            c.w += nxR

    assert (nxR+nxZ == nxV + sum(1 if c.w is not None else 0 for c in changes))
    return changes, nxR, nxZ


def _process_changes(changes, V, c, d, A, b, add_rows=False):
    c = copy.copy(c)
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
                    c = np.append(c, 0)
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
                c = np.append(c, -c[v])
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

    Bdok = dok_matrix((nrows, len(V)))
    # Collect the items from B
    for k,v in B.items():
        Bdok[k] = v
    # Merge in the items from A, shifting columns
    Adok = Acsc.todok()
    for k,v in Adok:
        #row, w = k
        #if w > V.nxR+V.nxZ:
        Bdok[k] = v
    return c, d, Adok.tocoo(), b


def convert_to_nonnegative_variables(ans, inequalities):
    #
    # Collect real variables that are changing
    #
    UxV = ans.U.x
    changes_U, nxR, nxZ = _find_nonpositive_variables(UxV, inequalities)
    UxV.resize(nxR, nxZ, UxV.nxB)
    UxV.lower_bounds = np.zeros(nxR+nxZ+UxV.nxB)
    changes_L = {}
    for i in range(len(ans.L)):
        LxV = ans.L[i].x
        changes_L_, nxR, nxZ = _find_nonpositive_variables(LxV, inequalities)
        LxV.resize(nxR, nxZ, LxV.nxB)
        LxV.lower_bounds = np.zeros(nxR+nxZ+LxV.nxB)
        changes_L[i] = changes_L_
    #
    # Process changes related to upper-level variables
    #
    if len(changes_U) > 0:
        UxV = ans.U.x
        #UcUxV = ans.U.c.U.x #getattr(ans.U.c.U, vstr)
        #UAUxV = ans.U.A.U.x #getattr(ans.U.A.U, vstr)
        ans.U.c.U.x, ans.U.d, ans.U.A.U.x, ans.U.b = \
                _process_changes(changes_U, UxV, ans.U.c.U.x, ans.U.d, ans.U.A.U.x, ans.U.b, add_rows=True)
        #setattr(ans.U.c.U, vstr, UcUxV)
        #setattr(ans.U.A.U, vstr, UAUxV)
        for i,L in enumerate(ans.L):
            #LcUxV = getattr(ans.L[i].c.U, vstr)
            #LAUxV = getattr(ans.L[i].A.U, vstr)
            ans.L[i].c.U.x, ans.L[i].d, ans.L[i].A.U.x, ans.L[i].b = \
                _process_changes(changes_U, UxV, ans.L[i].c.U.x, L.d, ans.L[i].A.U.x, L.b)
            #setattr(ans.L[i].c.U, vstr, LcUxV)
            #setattr(ans.L[i].A.U, vstr, LAUxV)
    #
    # Process changes related to lower-level variables
    #
    for i,L in enumerate(ans.L):
        LxV = ans.L[i].x
        if len(changes_L[i]) > 0:
            #UcLxV = getattr(ans.U.c.L[i], vstr)
            #UALxV = getattr(ans.U.A.L[i], vstr)
            ans.U.c.L[i].x, ans.U.d, ans.U.A.L[i].x, ans.U.b = \
                    _process_changes(changes_L[i], LxV, ans.U.c.L[i].x, ans.U.d, ans.U.A.L[i].x, ans.U.b)
            #setattr(ans.U.c.L[i], vstr, UcLxV)
            #setattr(ans.U.A.L[i], vstr, UALxV)
            #
            #LcLxV = getattr(ans.L[i].c.L[i], vstr)
            #LALxV = getattr(ans.L[i].A.L[i], vstr)
            ans.L[i].c.L[i].x, ans.L[i].d, ans.L[i].A.L[i].x, ans.L[i].b = \
                    _process_changes(changes_L[i], LxV, ans.L[i].c.L[i].x, ans.L[i].d, ans.L[i].A.L[i].x, ans.L[i].b, add_rows=True)
            #setattr(ans.L[i].c.L[i], vstr, LcLxV)
            #setattr(ans.L[i].A.L[i], vstr, LALxV)
    #
    # Resize constraint matrices
    #
    # After processing upper and lower variables, we may have added constraints.  The other
    # upper/lower constraint matrices need to be resized as well.
    #
    for i in range(len(ans.L)):
        if ans.U.A.L[i].x is not None:
            ans.U.A.L[i].x.resize( [len(ans.U.b), len(ans.L[i].x)] )
        if ans.L[i].A.U.x is not None:
            ans.L[i].A.U.x.resize( [len(ans.L[i].b), len(ans.U.x)] )
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


def convert_sense(level):
    level.minimize = True
    level.d *= -1
    if level.c.U.xR is not None:
        level.c.U.xR *= -1
    if level.c.U.xZ is not None:
        level.c.U.xZ *= -1
    if level.c.U.xB is not None:
        level.c.U.xB *= -1
    for i in range(len(level.c.L)):
        if level.c.L[i].xR is not None:
            level.c.L[i].xR *= -1
        if level.c.L[i].xZ is not None:
            level.c.L[i].xZ *= -1
        if level.c.L[i].xB is not None:
            level.c.L[i].xB *= -1


def convert_to_minimization(ans):
    if not ans.U.minimize:
        convert_sense(ans.U)
    for i in range(len(ans.L)):
        if not ans.L[i].minimize:
            convert_sense(ans.L[i])


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
    if inequalities:
        #
        # Creating inequality constraints from equalities by 
        # duplicating constraints
        #
        if not ans.U.inequalities:
            bnew = np.copy(ans.U.b)
            bnew *= -1
            ans.U.b = np.concatenate((ans.U.b, bnew))
            ans.U.A.U.xR = add_ineq_constraints(ans.U.A.U.xR)
            ans.U.A.U.xZ = add_ineq_constraints(ans.U.A.U.xZ)
            ans.U.A.U.xB = add_ineq_constraints(ans.U.A.U.xB)
            for i in range(len(ans.L)):
                ans.U.A.L[i].xR = add_ineq_constraints(ans.U.A.L[i].xR)
                ans.U.A.L[i].xZ = add_ineq_constraints(ans.U.A.L[i].xZ)
                ans.U.A.L[i].xB = add_ineq_constraints(ans.U.A.L[i].xB)
        for i in range(len(ans.L)):
            if not ans.L[i].inequalities:
                bnew = np.copy(ans.L[i].b)
                bnew *= -1
                ans.L[i].b = np.concatenate((ans.L[i].b, bnew))
                ans.L[i].A.U.xR = add_ineq_constraints(ans.L[i].A.U.xR)
                ans.L[i].A.U.xZ = add_ineq_constraints(ans.L[i].A.U.xZ)
                ans.L[i].A.U.xB = add_ineq_constraints(ans.L[i].A.U.xB)
                ans.L[i].A.L[i].xR = add_ineq_constraints(ans.L[i].A.L[i].xR)
                ans.L[i].A.L[i].xZ = add_ineq_constraints(ans.L[i].A.L[i].xZ)
                ans.L[i].A.L[i].xB = add_ineq_constraints(ans.L[i].A.L[i].xB)
    else:
        #
        # Add slack variables to create equality constraints from inequalities
        #
        if ans.U.inequalities:
            j = len(ans.U.xR)#-len(ans.U.b)
            ans.U.xR.resize( len(ans.U.xR) + len(ans.U.b), lb=0 )
            B = dok_matrix((len(ans.U.b), len(ans.U.xR)))
            for i in range(len(ans.U.b)):
                B[i,j] = 1
                j += 1
                if ans.U.c.U.xR is not None:
                    ans.U.c.U.xR = np.append(ans.U.c.U.xR, 0)
                for k,L in enumerate(ans.L):
                    if ans.L[k].c.U.xR is not None:
                        ans.L[k].c.U.xR = np.append(ans.L[k].c.U.xR, 0)
            ans.U.A.U.xR = combine_matrices(ans.U.A.U.xR, B)

        for i in range(len(ans.L)):
            if ans.L[i].A.U.xR is not None:
                ans.L[i].A.U.xR.resize( (ans.L[i].A.U.xR.shape[0], len(ans.U.xR)) )

            if ans.L[i].inequalities:
                j = len(ans.L[i].xR)#-len(ans.L[i].b)
                ans.L[i].xR.resize( len(ans.L[i].xR) + len(ans.L[i].b), lb=0 )
                B = dok_matrix((len(ans.L[i].b), len(ans.L[i].xR)))
                for k in range(len(ans.L[i].b)):
                    B[k,j] = 1
                    j += 1
                    if ans.U.c.L[i].xR is not None:
                        ans.U.c.L[i].xR = np.append(ans.U.c.L[i].xR, 0)
                    if ans.L[i].c.L[i].xR is not None:
                        ans.L[i].c.L[i].xR = np.append(ans.L[i].c.L[i].xR, 0)
                ans.L[i].A.L[i].xR = combine_matrices(ans.L[i].A.L[i].xR, B)

                if ans.U.A.L[i].xR is not None:
                    ans.U.A.L[i].xR.resize( (ans.U.A.L[i].xR.shape[0], len(ans.L[i].xR)) )
    #
    # Update inequality values
    #
    ans.U.inequalities = inequalities
    for i in range(len(ans.L)):
        ans.L[i].inequalities = inequalities


def get_multipliers(lbp, changes_U, changes_L, real=True):
    if real:
        vstr = 'xR'
    else:
        vstr = 'xZ'
    # 
    # If there were no changes, then the multiplier is 1
    #
    multipliers_U =   [[(i,1)] for i in getattr(lbp.U, vstr)]
    multipliers_L = [ [[(i,1)] for i in getattr(lbp.L[j], vstr)] for j in range(len(lbp.L)) ]
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
        for i in range(len(lbp.L)):
            if nxZ == 0:
                lbp.L[i].c.U.xZ = lbp.L[i].c.U.xB
                lbp.L[i].A.U.xZ = lbp.L[i].A.U.xB
            else:
                lbp.L[i].c.U.xZ = np.concatenate((lbp.L[i].c.U.xZ, lbp.L[i].c.U.xB))
                lbp.L[i].A.U.xZ = hstack([lbp.L[i].A.U.xZ, lbp.L[i].A.U.xB], format='csr')
            lbp.L[i].c.U.xB = None
            lbp.L[i].A.U.xB = None

    for i in range(len(lbp.L)):
        if len(lbp.L[i].xB) > 0:
            nxZ = len(lbp.L[i].xZ)
            nxB = len(lbp.L[i].xB)
            lbp.L[i].xZ.resize(nxZ+nxB, lb=0, ub=1)
            lbp.L[i].xB.resize(0)

            if nxZ == 0:
                lbp.U.c.L[i].xZ = lbp.U.c.L[i].xB
                lbp.U.A.L[i].xZ = lbp.U.A.L[i].xB
            else:
                lbp.U.c.L[i].xZ = np.concatenate((lbp.U.c.L[i].xZ, lbp.U.c.L[i].xB))
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
    multipliers_UxR, multipliers_LxR = get_multipliers(lbp, changes_UxR, changes_LxR, real=True)
    multipliers_UxZ, multipliers_LxZ = get_multipliers(lbp, changes_UxZ, changes_LxZ, real=False)

    return ans, LBP_SolutionManager( real=(multipliers_UxR, multipliers_LxR), 
                                     integer=(multipliers_UxZ, multipliers_LxZ) )

