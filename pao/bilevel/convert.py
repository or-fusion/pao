import itertools
import numpy as np
import pyomo.environ as pe
from pyomo.repn import generate_standard_repn
from pyomo.core.base import SortComponents, is_fixed

from pao.lbp import LinearBilevelProblem
from .components import SubModel


def _bound_equals(exp, num):
    if exp is None:
        return False
    if is_fixed(exp):
        return pe.value(exp) == num
    return False


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
        self.unfixedvars = set()
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
        # c.U
        #
        assert (len(self.orepn) <= 1), "PAO model has %d objectives specified, but a LinearBilevelProblem can have no more than one" % len(self.orepn)
        if len(self.orepn) == 1:
            level.d = pe.value(self.orepn[0].constant)

            c_xR = {}
            c_xZ = {}
            c_xB = {}
            for i in levelmap:
                c_xR[i] = {}
                c_xZ[i] = {}
                c_xB[i] = {}

            for i,c  in enumerate(self.orepn[0].linear_coefs):
                vid = id(self.orepn[0].linear_vars[i])
                t, nid, j = vidmap[vid]
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
                    
            

def collect_multilevel_tree(block, var, vidmap={}, sortOrder=SortComponents.unsorted, fixed=set()):
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
    fixedvars = fixedvars | curr.fixedvars
    curr.children = \
        [collect_multilevel_tree(submodel, var, vidmap, fixed=fixedvars) 
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
        curr.orepn.append( repn )
    #
    # Constraints
    #
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
        curr.crepn.append( repn )

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
    for repn in itertools.chain(curr.orepn, curr.crepn):
        for v in repn.linear_vars:
            i = id(v)
            if i not in knownvars:
                curr.unfixedvars.add(i)
                var[i] = v
        for v,w in repn.quadratic_vars:
            i = id(v)
            if i not in knownvars:
                curr.unfixedvars.add(i)
                var[i] = v
            i = id(w)
            if i not in knownvars:
                curr.unfixedvars.add(i)
                var[i] = v
    #
    # Categorize variables
    #
    if sortOrder == SortComponents.unsorted:
        for i,v in var.items():
            curr.categorize_variable(i,v, vidmap)
    else:
        for i,_,v in sorted(((i,v.name,v) for i,v in var.items()), key=lambda arg: arg[1]):
            curr.categorize_variable(i,v, vidmap)
    #
    # Return root of this tree
    #
    return curr


def convert_pyomo2LinearBilevelProblem1(model, determinism=1, inequalities=True):
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
    tree = collect_multilevel_tree(model, var, vidmap, sortOrder=sortOrder)
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

    return M
    

# WEH - I suspect we'll try out multiple conversion functions, but this will be the default function
convert_pyomo2LinearBilevelProblem = convert_pyomo2LinearBilevelProblem1

