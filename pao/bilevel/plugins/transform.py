#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
pao.bilevel.plugins.transform

Definition of a base class for bilevel transformation.
"""

from pyomo.core import Transformation, Var, ComponentUID
from ..components import SubModel


class BaseBilevelTransformation(Transformation):
    """
    Base class defining methods commonly used to transform
    bilevel programs.
    """

    def _preprocess(self, tname, instance, sub=None):
        """
        Iterate over the model collecting variable data,
        until the submodel is found.
        """
        var = {}
        submodel = None
        for (name, data) in instance.component_map(active=True).items():
            if isinstance(data, Var):
                var[name] = data
            elif isinstance(data, SubModel):
                if sub is None or sub == name:
                    sub = name
                    submodel = data
                    break
        if submodel is None:
            raise RuntimeError("Missing submodel: "+str(sub))
        #
        instance._transformation_data[tname].submodel = [name]
        #
        # Fix variables
        #
        if submodel._fixed:
            fixed = []
            unfixed = []
            for i in submodel._fixed:
                name = i.name
                fixed.append(name)
            for v in var:
                if not v in fixed:
                    unfixed.append((v, getattr(submodel._parent(), v).is_indexed()))
        elif submodel._var:
            fixed = []
            unfixed = [(v.name, v.is_indexed()) for v in submodel._var]
            unfixed_names = [v.name for v in submodel._var]
            for v in var:
                if not v in unfixed_names:
                    fixed.append(v)
        else:
            raise RuntimeError("Must specify 'fixed' or 'unfixed' options")
        #
        self._submodel = sub
        self._upper_vars = var
        self._fixed_upper_vars = fixed
        self._unfixed_upper_vars = unfixed
        instance._transformation_data[tname].fixed = [ComponentUID(var[v]) for v in fixed]
        return submodel

    def _fix_all(self):
        """
        Fix the upper variables
        """
        self._fixed_cache = {}
        for v in self._fixed_upper_vars:
            self._fixed_cache[v] = self._fix(self._upper_vars[v])

    def _unfix_all(self):
        """
        Unfix the upper variables
        """
        for v in self._fixed_upper_vars:
            self._unfix(self._upper_vars[v], self._fixed_cache[v])

    def _fix(self, var):
        """
        Fix the upper level variables, tracking the variables that were
        modified.
        """
        cache = []
        for i, vardata in var.items():
            if not vardata.fixed:
                vardata.fix()
                cache.append(i)
        return cache

    def _unfix(self, var, cache):
        """
        Unfix the upper level variables.
        """
        for i in cache:
            var[i].unfix()
