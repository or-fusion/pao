import itertools
import pyomo.environ as pe
from pyomo.core.base import SortComponents
from pao.tensor import LinearBilevelProblem
from .components import SubModel


def collect_multilevel_tree(model, var, sortOrder=SortComponents.unsorted):
    """
    Traverse the model and generate a tree of the SubModel components
    """
    class Node(object):

        def __init__(self, node):
            self.node = node
            self.children = []
            self.fixed = set()
            #
            # Collect vardata that are declared fixed.  If a variable component
            # is specified, then collect all of its vardata objects.
            #
            for v in getattr(node, '_fixed', []):
                if v.is_indexed():
                    for vardata in v.values():
                        self.fixed.add(id(vardata))
                else:
                    self.fixed.add(id(v))

    curr = Node(model)
    #
    # Collect Var and SubModel components
    #
    # TODO: Others?  Throw an exception is unexpected components are found?
    #
    submodels = []
    for data in model.component_objects(active=True, descend_into=True, sort=sortOrder):
        name = data.name
        if isinstance(data, pe.Var):
            var[id(data)] = (data,len(var))
        elif isinstance(data, SubModel):
            submodels.append(data)
    #
    # Recurse
    #
    curr.children = [collect_multilevel_tree(node, var) for node in submodels]
    #
    # Return root of this tree
    #
    return curr


def collect_linear(submodel, level, fixed=[]):
    """
    Traverse the Pyomo model collecting linear constraints.  Add these to the 
    LinearBilevelProblem level representation.
    """
    # 
    for odata in submodel.component_objects(pe.Objective, active=True):
        if odata.is_indexed():
            for _name, _odata in odata.items():
                # COLLECT
                pass
        else:
            pass
            # COLLECT
    #
    for cdata in submodel.component_objects(pe.Constraint, active=True, descend_into=True):
        if cdata.is_indexed():
            for _name, _cdata in cdata.items():
                # COLLECT
                pass
        else:
            pass
            # COLLECT


def convert_pyomo2LinearBilevelProblem1(model, determinism=1):
    """
    Traverse the model an generate a LinearBilevelProblem.  Generate errors
    if this problem cannot be represented in this form.

    This conversion applies the following transformations:
        * replaces quadratic terms, x*y, where x or y is integer and both x and y are bounded
    """
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
    tree = collect_multilevel_tree(model, var, sortOrder=sortOrder)
    for key,v in var.items():
        print(key,v[0],v[1])
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
    M = LinearBilevelProblem()
    U = M.add_upper()
    collect_linear(tree.node, U)
    for i,c in enumerate(tree.children):
        L = M.add_lower()
        collect_linear(c.node, L)
    

# WEH - I suspect we'll try out multiple conversion functions, but this will be the default function
convert_pyomo2LinearBilevelProblem = convert_pyomo2LinearBilevelProblem1

