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
            self._fixed_vardata = [vardata for v in submodel._fixed for vardata in v.values()]
        else:
            raise RuntimeError("Must specify 'fixed' or 'unfixed' options")
        #
        self._submodel = sub
        instance._transformation_data[tname].fixed = [ComponentUID(v) for v in self._fixed_vardata]
        self._fixed_ids = set()
        return submodel

    def _fix_all(self):
        """
        Fix the upper variables
        """
        for vardata in self._fixed_vardata:
            if not vardata.fixed:
                self._fixed_ids.add(id(vardata))
                vardata.fixed = True

    def _unfix_all(self):
        """
        Unfix the upper variables
        """
        for vardata in self._fixed_vardata:
            if id(vardata) in self._fixed_ids:
                vardata.fixed = False
                self._fixed_ids.remove(id(vardata))
