import copy
import itertools
import numpy as np
import pyomo.environ as pe
from pyomo.repn import generate_standard_repn
from pyomo.core.base import SortComponents, is_fixed
from scipy.sparse import dok_matrix

from pao.lbp import LinearMultilevelProblem
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
            # value equals, then we are assuming the LinearMultilevelProblem will
            # be re-generated if those parameter values change.
            #
            if _bound_equals(v.lb, 0) and _bound_equals(v.ub, 1):
                vidmap[vid] = (2, self.nid, len(self.xB))
                self.xB[len(self.xB)] = vid
            else:
                vidmap[vid] = (1, self.nid, len(self.xZ))
                self.xZ[len(self.xZ)] = vid

        else:
            assert (v.is_continuous()), "Variable '%s' has a domain type that is not continuous, integer or binary" % str(v)
            vidmap[vid] = (0, self.nid, len(self.xR))
            self.xR[len(self.xR)] = vid

    def add_levels(self, L, levelmap, treemap):
        treemap[self.nid] = self
        levelmap[L.id] = L
        for child in self.children:
            L_ = L.add_lower(id=child.nid)
            child.add_levels(L_, levelmap, treemap)

    def initialize_level_vars(self, level, inequalities, var):
        """
        Initialize the level object...
        """
        level.x._resize(nxR=len(self.xR), nxZ=len(self.xZ), nxB=len(self.xB))
        #
        # xR
        #
        for i in self.xR:
            vid = self.xR[i]
            val = var[vid].lb
            if val is not None:
                level.x.lower_bounds[i] = val
            val = var[vid].ub
            if val is not None:
                level.x.upper_bounds[i] = val
        #
        # xZ
        #
        for i in self.xZ:
            vid = self.xZ[i]
            val = var[vid].lb
            if val is not None:
                level.x.lower_bounds[i+level.x.nxR] = val
            val = var[vid].ub
            if val is not None:
                level.x.upper_bounds[i+level.x.nxR] = val
        #
        # xB
        #
        for i in self.xB:
            level.x.lower_bounds[i+level.x.nxR+level.x.nxZ] = 0
            level.x.upper_bounds[i+level.x.nxR+level.x.nxZ] = 1

    def initialize_level(self, level, inequalities, var, vidmap, levelmap):
        #
        # c
        #
        assert (len(self.orepn) <= 1), "PAO model has %d objectives specified, but a LinearMultilevelProblem can have no more than one" % len(self.orepn)
        if len(self.orepn) == 1:
            repn = self.orepn[0][0]
            if self.orepn[0][1] == pe.maximize:
                level.minimize = False
            level.d = pe.value(repn.constant)

            c = {}
            for j in levelmap:
                node = levelmap[j]
                c[j] = [0]*node.x.num

            for i,val  in enumerate(repn.linear_coefs):
                vid = id(repn.linear_vars[i])
                t, nid, j = vidmap[vid]
                L = levelmap[nid]
                if t == 0:
                    c[nid][j] = pe.value(val)
                elif t == 1:
                    c[nid][j+L.x.nxR] = pe.value(val)
                elif t == 2:
                    c[nid][j+L.x.nxR+L.x.nxZ] = pe.value(val)

            for j in levelmap:
                for v in c[j]:
                    if v != 0:
                        level.c[j] = c[j]
                        break
        #
        # A
        #
        if len(self.crepn) > 0:
            A = {}
            for i in levelmap:
                A[i] = {}
            b = []
            nrows = len(self.crepn)

            for k in range(len(self.crepn)):
                repn = self.crepn[k][0]
                for i,c  in enumerate(repn.linear_coefs):
                    vid = id(repn.linear_vars[i])
                    t, nid, j = vidmap[vid]
                    L = levelmap[nid]
                    c_ = pe.value(c)
                    if c_ != 0:
                        if t == 0:
                            A[nid][k,j] = c_
                        elif t == 1:
                            A[nid][k,j+L.x.nxR] = c_
                        elif t == 2:
                            A[nid][k,j+L.x.nxR+L.x.nxZ] = c_
                b.append(self.crepn[k][1] - repn.constant)

            level.b = b

            for j in levelmap:
                L = levelmap[j]
                if len(A[j]) > 0:
                    mat = dok_matrix((nrows, L.x.num))
                    for key, val in A[j].items():
                        mat[key] = val
                    level.A[L.id] = mat
            

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
                U = lbp.U
                if t == 0:
                    v.value = lbp.U.x.values[j]
                elif t == 1:
                    v.value = lbp.U.x.values[j+U.x.nxR]
                elif t == 2:
                    v.value = lbp.U.x.values[j+U.x.nxR+U.x.nxZ]
            else:
                L = lbp.U.LL[nid-1]
                if t == 0:
                    v.value = lbp.U.LL[nid-1].x.values[j]
                elif t == 1:
                    v.value = lbp.U.LL[nid-1].x.values[j+L.x.nxR]
                elif t == 2:
                    v.value = lbp.U.LL[nid-1].x.values[j+L.x.nxR+L.x.nxZ]


def convert_pyomo2LinearMultilevelProblem1(model, *, determinism=1, inequalities=True):
    """
    Traverse the model an generate a LinearMultilevelProblem.  Generate errors
    if this problem cannot be represented in this form.

    This conversion applies the following transformations:
        * replaces quadratic terms, x*y, where x or y is integer and both x and y are bounded

    Args
    ---- 
    model
        A Pyomo model object.
    determinism: int
        Indicates whether the traversal of **model** is
        ordered.  Valid values are:

                * 0 - Unordered traversal of **model**
                * 1 - Ordered traversal of component indices in **model**
                * 2 - Ordered traversal of components by name in **model**

    inequalities: bool
        If True, then the LinearMultilevelProblem object represents all
        constraints as less-than-or-equal inequalities.  Otherwise,
        the LinearMultilevelProblem represents all constraints as equalities.

    Returns
    -------
    LinearMultilevelProblem
        This object corresponds to the problem in **model**.
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
    # Collect SubModel representations
    #
    treemap = {}
    levelmap = {}
    M = LinearMultilevelProblem()
    U = M.add_upper(id=tree.nid)
    treemap[tree.nid] = tree

    tree.add_levels(U, levelmap, treemap)
    
    for L in M.levels():
        node = treemap[L.id]
        node.initialize_level_vars(L, inequalities, var)
    
    for L in M.levels():
        node = treemap[L.id]
        node.initialize_level(L, inequalities, var, vidmap, levelmap)
    #
    # Cleanup global memory
    #
    Node.global_list = []

    return M, PyomoSubmodel_SolutionManager_LBP(var, vidmap)
    

# WEH - I suspect we'll try out multiple conversion functions, but this will be the default function
convert_pyomo2LinearMultilevelProblem = convert_pyomo2LinearMultilevelProblem1
convert_pyomo2lbp = convert_pyomo2LinearMultilevelProblem1

