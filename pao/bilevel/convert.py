import copy
import itertools
import numpy as np
import pyomo.environ as pe
from pyomo.repn import generate_standard_repn
from pyomo.core.base import SortComponents, is_fixed
from scipy.sparse import dok_matrix

from pao.lbp import LinearBilevelProblem
from .components import SubModel


def _bound_equals(exp, num):
    if exp is None:
        return False
    if is_fixed(exp):
        return pe.value(exp) == num
    return False                        # pragma: no cover
                                        # WEH - will we ever reach this point?  What if the LB is a mutable parameter?


class Node(object):

    global_list = []

    def __init__(self, node):
        self.nid = len(Node.global_list)
        Node.global_list.append(self)
        self.node = node
        self.children = []
        self.orepn = []
        self.crepn = []
        self.fixedvars = set()
        self.unfixedvars = set()    # unfixed variables used in expressions
        self.xR = {}
        self.xZ = {}
        self.xB = {}
        #
        # Collect vardata that are declared fixed.  If a variable component
        # is specified, then collect all of its vardata objects.
        #
        for v in getattr(node, '_fixed', []):
            if v.is_indexed():
                for vardata in v.values():
                    self.fixedvars.add(id(vardata))
            else:
                self.fixedvars.add(id(v))

    def categorize_variable(self, vid, v, vidmap):
        if v.is_binary():
            vidmap[vid] = (2, self.nid, len(self.xB))
            self.xB[len(self.xB)] = vid

        elif v.is_integer():
            #
            # The _bounds_equals function only returns True if the 
            # bound is a fixed value and its value equals the 2nd argument.
            # If the bound value is specified with Pyomo Parameters and their
            # value equals, then we are assuming the LinearBilevelProblem will
            # be re-generated if those parameter values change.
            #
            if _bound_equals(v.lb, 0) and _bound_equals(v.ub, 1):
                vidmap[vid] = (2, self.nid, len(self.xB))
                self.xB[len(self.xB)] = vid
            else:
                vidmap[vid] = (1, self.nid, len(self.xZ))
                #print(len(self.xZ), self.nid, vid)
                self.xZ[len(self.xZ)] = vid

        else:
            assert (v.is_continuous()), "Variable '%s' has a domain type that is not continuous, integer or binary" % str(v)
            vidmap[vid] = (0, self.nid, len(self.xR))
            self.xR[len(self.xR)] = vid

    def initialize_level_vars(self, level, inequalities, var):
        """
        Initialize the level object...
        """
        #
        # xR
        #
        if len(self.xR) > 0:
            level.xR.resize(len(self.xR))
            lb = [np.NINF]*len(self.xR)
            ub = [np.PINF]*len(self.xR)
            for i in self.xR:
                vid = self.xR[i]
                val = var[vid].lb
                if val is not None:
                    lb[i] = val
                val = var[vid].ub
                if val is not None:
                    ub[i] = val
            level.xR.lower_bounds = lb
            level.xR.upper_bounds = ub
        #
        # xZ
        #
        if len(self.xZ) > 0:
            level.xZ.resize(len(self.xZ))
            lb = [np.NINF]*len(self.xZ)
            ub = [np.PINF]*len(self.xZ)
            for i in self.xZ:
                vid = self.xZ[i]
                val = var[vid].lb
                if val is not None:
                    lb[i] = val
                val = var[vid].ub
                if val is not None:
                    ub[i] = val
            level.xZ.lower_bounds = lb
            level.xZ.upper_bounds = ub
        #
        # xB
        #
        level.xB.resize(len(self.xB))

    def initialize_level(self, level, inequalities, var, vidmap, levelmap):
        #
        # c
        #
        assert (len(self.orepn) <= 1), "PAO model has %d objectives specified, but a LinearBilevelProblem can have no more than one" % len(self.orepn)
        if len(self.orepn) == 1:
            repn = self.orepn[0][0]
            if self.orepn[0][1] == pe.maximize:
                level.minimize = False
            level.d = pe.value(repn.constant)

            c_xR = {}
            c_xZ = {}
            c_xB = {}
            for i in levelmap:
                c_xR[i] = {}
                c_xZ[i] = {}
                c_xB[i] = {}

            for i,c  in enumerate(repn.linear_coefs):
                vid = id(repn.linear_vars[i])
                t, nid, j = vidmap[vid]
                #print(t,nid,j)
                if t == 0:
                    c_xR[nid][j] = pe.value(c)
                elif t == 1:
                    c_xZ[nid][j] = pe.value(c)
                elif t == 2:
                    c_xB[nid][j] = pe.value(c)

            for j in levelmap:
                node = levelmap[j]
                if len(c_xR[j]) > 0:
                    tmp = [0]*len(node.xR)
                    for i,v in c_xR[j].items():
                        tmp[i] = v
                    if j == 0:
                        level.c.U.xR = tmp
                    else:
                        level.c.L[j-1].xR = tmp
                if len(c_xZ[j]) > 0:
                    tmp = [0]*len(node.xZ)
                    for i,v in c_xZ[j].items():
                        tmp[i] = v
                    if j == 0:
                        level.c.U.xZ = tmp
                    else:
                        level.c.L[j-1].xZ = tmp
                if len(c_xB[j]) > 0:
                    tmp = [0]*len(node.xB)
                    for i,v in c_xB[j].items():
                        tmp[i] = v
                    if j == 0:
                        level.c.U.xB = tmp
                    else:
                        level.c.L[j-1].xB = tmp
        #
        # A
        #
        if len(self.crepn) > 0:
            A_xR = {}
            A_xZ = {}
            A_xB = {}
            for i in levelmap:
                A_xR[i] = {}
                A_xZ[i] = {}
                A_xB[i] = {}
            b = []
            nrows = len(self.crepn)

            for k in range(len(self.crepn)):
                repn = self.crepn[k][0]
                for i,c  in enumerate(repn.linear_coefs):
                    vid = id(repn.linear_vars[i])
                    t, nid, j = vidmap[vid]
                    if t == 0:
                        A_xR[nid][k,j] = pe.value(c)
                    elif t == 1:
                        A_xZ[nid][k,j] = pe.value(c)
                    elif t == 2:
                        A_xB[nid][k,j] = pe.value(c)
                b.append(self.crepn[k][1] - repn.constant)

            level.b = b

            for j in levelmap:
                node = levelmap[j]
                if len(A_xR[j]) > 0:
                    mat = dok_matrix((nrows, len(node.xR)))
                    for key, val in A_xR[j].items():
                        mat[key] = val
                    if j == 0:
                        level.A.U.xR = mat
                    else:
                        level.A.L[j-1].xR = mat
                if len(A_xZ[j]) > 0:
                    mat = dok_matrix((nrows, len(node.xZ)))
                    for key, val in A_xZ[j].items():
                        mat[key] = val
                    if j == 0:
                        level.A.U.xZ = mat
                    else:
                        level.A.L[j-1].xZ = mat
                if len(A_xB[j]) > 0:
                    mat = dok_matrix((nrows, len(node.xB)))
                    for key, val in A_xB[j].items():
                        mat[key] = val
                    if j == 0:
                        level.A.U.xB = mat
                    else:
                        level.A.L[j-1].xB = mat
                
            

def negate_repn(repn):
    trepn = copy.copy(repn)
    trepn.constant *= -1
    trepn.linear_coefs = list(-1*repn.linear_coefs[i] for i in range(len(repn.linear_coefs)))
    trepn.linear_vars = list(repn.linear_vars)
    return trepn


def collect_multilevel_tree(block, var, vidmap={}, sortOrder=SortComponents.unsorted, fixed=set(), inequalities=None):
    """
    Traverse the model and generate a tree of the SubModel components
    """
    #
    # Roof of the current subtree, defined by the block
    #
    curr = Node(block)
    #
    # Recurse, collecting Submodel components
    #
    fixedvars = fixed | curr.fixedvars
    curr.children = \
        [collect_multilevel_tree(submodel, var, vidmap, fixed=fixedvars, inequalities=inequalities) 
         for submodel in block.component_objects(SubModel, active=True, descend_into=True, sort=sortOrder)]
    #
    # Collect objectives and constraints in the current submodel.
    # Note that we do not recurse into SubModel blocks.
    #
    # Objectives
    #
    for odata in block.component_data_objects(pe.Objective, active=True, sort=sortOrder, descend_into=True):
        repn = generate_standard_repn(odata.expr)
        assert (repn.is_linear()), "Objective '%s' has a body with nonlinear terms" % odata.name
        degree = repn.polynomial_degree()
        if degree == 0:
            continue # trivial, so skip
        curr.orepn.append( (repn, odata.sense) )
    #
    # Constraints
    #
    block.zzz_PAO_SlackVariables = pe.VarList(domain=pe.NonNegativeReals)
    for cdata in block.component_data_objects(pe.Constraint, active=True, sort=sortOrder, descend_into=True):
        if (not cdata.has_lb()) and (not cdata.has_ub()):
            assert not cdata.equality, "Constraint '%s' is an equality with an infinite right-hand-side" % cdata.name
            # non-binding, so skip
            continue                            # pragma: no cover
        repn = generate_standard_repn(cdata.body)
        assert (repn.is_linear()), "Constraint '%s' has a body with nonlinear terms" % cdata.name
        degree = repn.polynomial_degree()
        if degree == 0:
            if cdata.equality:
                assert pe.value(cdata.body) == pe.value(cdata.lower), "Constraint '%s' is constant but it is not satisfied (equality)" % cdata.name
            else:
                if not cdata.lower is None:
                    assert pe.value(cdata.body) >= pe.value(cdata.lower), "Constraint '%s' is constant but it is not satisfied (lower-bound)" % cdata.name
                if not cdata.upper is None:
                    assert pe.value(cdata.body) <= pe.value(cdata.upper), "Constraint '%s' is constant but it is not satisfied (upper-bound)" % cdata.name
            # trivial, so skip
            continue                            # pragma: no cover
        else:
            if inequalities:
                if cdata.equality:
                    val = pe.value(cdata.lower)
                    curr.crepn.append( (repn, val) )
                    curr.crepn.append( (negate_repn(repn), -val) )
                else:
                    if cdata.lower is None and cdata.upper is None:             #pragma: no cover
                        # unbounded constraint
                        continue
                    if cdata.lower is not None:
                        curr.crepn.append( (negate_repn(repn), -pe.value(cdata.lower)) )
                    if cdata.upper is not None:
                        curr.crepn.append( (repn, pe.value(cdata.upper)) )
            else:
                if cdata.equality:
                    curr.crepn.append( (repn, pe.value(cdata.lower)) )
                else:
                    if cdata.lower is None and cdata.upper is None:             #pragma: no cover
                        # unbounded constraint
                        continue
                    if cdata.lower is not None:
                        trepn = negate_repn(repn)
                        trepn.linear_coefs.append(1)
                        trepn.linear_vars.append(block.zzz_PAO_SlackVariables.add())
                        curr.crepn.append( (trepn, -pe.value(cdata.lower)) )
                    if cdata.upper is not None:
                        repn.linear_coefs = list(repn.linear_coefs)
                        repn.linear_vars = list(repn.linear_vars)
                        repn.linear_coefs.append(1)
                        repn.linear_vars.append(block.zzz_PAO_SlackVariables.add())
                        curr.crepn.append( (repn, pe.value(cdata.upper)) )
    #
    # Collect the variables used by the children
    #
    childvars = set()
    for child in curr.children:
        childvars |= child.unfixedvars
    #
    # Add ids for variables in this block that have not been specified 
    # as fixed and which are not used in submodels
    #
    knownvars = curr.fixedvars | childvars
    newvars = []
    for repn in itertools.chain(curr.orepn, curr.crepn):
        for v in repn[0].linear_vars:
            i = id(v)
            if i not in knownvars:
                curr.unfixedvars.add(i)
                var[i] = v
                newvars.append(i)
                knownvars.add(i)
        if False:                                   # pragma: no cover
            for v,w in repn[0].quadratic_vars:
                i = id(v)
                if i not in knownvars:
                    curr.unfixedvars.add(i)
                    var[i] = v
                    newvars.append(i)
                    knownvars.add(i)
                i = id(w)
                if i not in knownvars:
                    curr.unfixedvars.add(i)
                    var[i] = v
                    newvars.append(i)
                    knownvars.add(i)
    #print("NID", curr.nid)
    #print("Fixed", len(curr.fixedvars))
    #print("Unfixed", len(curr.unfixedvars))
    #print("Child Unfixed", len(childvars))
    #
    # Categorize the new variables that were found
    #
    if sortOrder == SortComponents.unsorted:
        for i in newvars:
            curr.categorize_variable(i,var[i], vidmap)
    else:
        for k,_,w in sorted(((i,var[i].name,var[i]) for i in newvars), key=lambda arg: arg[1]):
                curr.categorize_variable(k,w, vidmap)
    #
    # Return root of this tree
    #
    return curr


class PyomoSubmodel_SolutionManager_LBP(object):

    def __init__(self, var, vidmap):
        self.var = var
        self.vidmap = vidmap

    def copy_from_to(self, *, lbp, pyomo):
        for vid in self.vidmap:
            v = self.var[vid]
            t, nid, j = self.vidmap[vid]
            if nid == 0:
                if t == 0:
                    v.value = lbp.U.xR.values[j]
                elif t == 1:
                    v.value = lbp.U.xZ.values[j]
                elif t == 2:
                    v.value = lbp.U.xB.values[j]
            else:
                if t == 0:
                    v.value = lbp.L[nid-1].xR.values[j]
                elif t == 1:
                    v.value = lbp.L[nid-1].xZ.values[j]
                elif t == 2:
                    v.value = lbp.L[nid-1].xB.values[j]


def convert_pyomo2LinearBilevelProblem1(model, *, determinism=1, inequalities=True):
    """
    Traverse the model an generate a LinearBilevelProblem.  Generate errors
    if this problem cannot be represented in this form.

    This conversion applies the following transformations:
        * replaces quadratic terms, x*y, where x or y is integer and both x and y are bounded
    """
    #
    # Cleanup global memory
    #
    Node.global_list = []
    #
    # Define sort order for searching Pyomo model
    #
    sortOrder = SortComponents.unsorted
    if determinism >= 1:
        sortOrder = sortOrder | SortComponents.indices
        if determinism >= 2:
            sortOrder = sortOrder | SortComponents.alphabetical
    #
    # Collect tree of bilevel models
    #
    var = {}
    vidmap = {}
    tree = collect_multilevel_tree(model, var, vidmap, sortOrder=sortOrder, inequalities=inequalities)
    #
    # We must have a least one SubModel
    #
    assert (len(tree.children) > 0), "Pyomo problem does not contain SubModel components"
    #
    # LinearBilevelProblem cannot represent problems with nested submodels (e.g. tri-level)
    #
    for i,child in enumerate(tree.children):
        assert (len(child.children) == 0), "Pyomo problem contains nested SubModel components"
    #
    # Collect SubModel representations
    #
    levelmap = {}
    M = LinearBilevelProblem()
    U = M.add_upper()
    levelmap[0] = U 
    tree.initialize_level_vars(U, inequalities, var)
    for i,c in enumerate(tree.children):
        L = M.add_lower()
        levelmap[i+1] = L[i]
        c.initialize_level_vars(L[i], inequalities, var)
    
    tree.initialize_level(U, inequalities, var, vidmap, levelmap)
    for i,c in enumerate(tree.children):
        c.initialize_level(L[i], inequalities, var, vidmap, levelmap)
    #
    # Cleanup global memory
    #
    Node.global_list = []

    return M, PyomoSubmodel_SolutionManager_LBP(var, vidmap)
    

# WEH - I suspect we'll try out multiple conversion functions, but this will be the default function
convert_pyomo2LinearBilevelProblem = convert_pyomo2LinearBilevelProblem1
convert_pyomo2lbp = convert_pyomo2LinearBilevelProblem1

