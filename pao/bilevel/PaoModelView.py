
__all__ = ['PoekModel', 'get_poekvardata']

import collections.abc
from pyomo.core.base import ConcreteModel, minimize, Set, Objective, Constraint
from pyomo.core.base import ObjectiveList, ConstraintList, _ObjectiveData, _ConstraintData, _VarData, VarList, Var


class X_VarDataSequence(collections.abc.Sequence):

    def __init__(self, poek_model):
        self.poek_model = poek_model
        self.data = []
        #
        # Initialize all of the standard repns.  This creates the list
        # of all VarData objects, which Pyomo assumes exist
        #
        vars_ = set()
        for i in range(poek_model.num_objectives()):
            expr = poek_model.get_objective(i)
            n = expr.repn_nlinear_vars()
            for j in range(n):
                vptr = expr.repn_linear_var(j)
                if vptr not in vars_:
                    vars_.add(vptr)
                    self.data.append( get_poekvardata(vptr) )
        for i in range(poek_model.num_constraints()):
            expr = poek_model.get_constraint(i).body
            n = expr.repn_nlinear_vars()
            for j in range(n):
                vptr = expr.repn_linear_var(j)
                if vptr not in vars_:
                    vars_.add(vptr)
                    self.data.append( get_poekvardata(vptr) )
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def values(self):
        for vptr in self.data:
            yield vptr

    def get(self, i, default):
        try:
            return self.__getitem__(i)
        except IndexError:
            return default


class X_ObjectiveDataSequence(collections.abc.Sequence):

    def __init__(self, poek_model):
        self.poek_model = poek_model
        self.cache = {}

    def __len__(self):
        return self.poek_model.num_objectives()

    def __getitem__(self, i):
        try:
            return self.cache[i]
        except:
            tmp = self.cache[i] = Poek_ObjectiveData( self.poek_model.get_objective(i) )  # minimize = True
            return tmp

    def get(self, i, default):
        try:
            return self.__getitem__(i)
        except IndexError:
            return default


class ObjectiveView(ObjectiveList):
    """
    An objective component that represents a list of objectives from
    another Pyomo model.
    """

    End             = (1003,)

    def __init__(self, pyomo_model):
        """Constructor"""
        args = (Set(),)
        kwargs = {}
        self.pyomo_model = pyomo_model
        Objective.__init__(self, *args, **kwargs)
        #self._data = ObjectiveDataSequence( poek_model )
        self._data = []

    def construct(self, data=None):
        if self._constructed:
            return
        self._data = [odata for odata in self.component_data_objects(Objective, active=True)
        for i in range(len(self._data)):
            self._index.add(i)
        self._constructed = True
        

class X_ConstraintDataSequence(collections.abc.Sequence):

    def __init__(self, poek_model):
        self.poek_model = poek_model
        self.cache = {}

    def __len__(self):
        return self.poek_model.num_constraints()

    def __getitem__(self, i):
        try:
            return self.cache[i]
        except:
            tmp = self.cache[i] = Poek_ConstraintData( self.poek_model.get_constraint(i) )
            return tmp

    def get(self, i, default):
        try:
            return self.__getitem__(i)
        except IndexError:
            return default


class ConstraintView(ConstraintList):
    """
    An constraint component that represents a list of constraints from
    another Pyomo model.
    """

    End             = (1003,)

    def __init__(self, pyomo_model):
        """Constructor"""
        args = (Set(),)
        kwargs = {}
        self.pyomo_model = pyomo_model
        Constraint.__init__(self, *args, **kwargs)
        #self._data = ConstraintDataSequence( poek_model )
        self._data = []

    def construct(self, data=None):
        if self._constructed:
            return
        self._data = [cdata for cdata in self.component_data_objects(Constraint, active=True)
        for i in range(len(self._data)):
            self._index.add(i)
        self._constructed = True
        

class IndexedComponentView(IndexedComponent):
    """
    A component that represents a list of variables from
    another Pyomo model.
    """

    End             = (1003,)

    def __init__(self, pyomo_model, ctype):
        """Constructor"""
        args = (Set(),)
        kwargs = {'ctype': ctype}
        IndexedComponent.__init__(self, *args, **kwargs)
        self.pyomo_model = pyomo_model
        self._data = []
        self.ctype = ctype

    def construct(self, data=None):
        if self._constructed:
            return
        self._data = [vardata for vardata in self.component_data_objects(self.ctype, active=True)
        for i in range(len(self._data)):
            self._index.add(i)
        self._constructed = True
        

class PAOModelView(ConcreteModel):
    """
    A concrete optimization model that creates a flat
    representation of a Pyomo model with nested SubModels

    NOTE: This assumes the model is not abstract.
    """

    def __init__(self, pyomo_model, **kwds):
        kwds['concrete'] = True
        self.pyomo_model = pyomo_model
        ConcreteModel.__init__(self, tuple(), **kwds)
        self.variable        = IndexedComponentView(pyomo_model, Var)
        self.objective       = IndexedComponentView(pyomo_model, Objective)
        self.constraint      = IndexedComponentView(pyomo_model, Constraint)
        ##self.submodel        = IndexedComponentView(pyomo_model, SubModel)
        #self.complementarity = IndexedComponentView(pyomo_model, Complementarity)
        #GDP
        #MPEC
        #DAE
        #PYSP

