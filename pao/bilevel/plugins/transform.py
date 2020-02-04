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

from pyomo.core import Transformation, Var, ComponentUID, Block
from ..components import SubModel
import logging

logger = logging.getLogger(__name__)

class BaseBilevelTransformation(Transformation):
    """
    Base class defining methods commonly used to transform
    bilevel programs.
    """
    _fixed_vardata = dict()
    _fixed_ids = set()
    _submodel = dict()

    @property
    def fixed_vardata(self):
        return self._fixed_vardata

    @fixed_vardata.setter
    def fixed_vardata(self,key,val):
        self._fixed_vardata[key] = val

    @property
    def submodel(self):
        return self._submodel

    @submodel.setter
    def submodel(self,key,val):
        self._submodel[key] = val

    def _nest_level(self,block,level=2):
        """
        Determines depth level of block in model hierarchy

        """
        if block.parent_block() == block.root_block():
            return level
        else:
            level += 1
            self._nest_level(block.parent_block(),level)

    def _preprocess(self, tname, instance):
        """
        Iterate over the model collecting variable data,
        until all submodels are found.

        """
        var = {}
        for (name, data) in instance.component_map(active=True).items():
            if isinstance(data, Var):
                var[name] = data
            elif isinstance(data, SubModel):
                submodel = data
                if submodel is None:
                    e = "Missing submodel: "+str(name)
                    logger.error(e)
                    raise RuntimeError(e)
                instance._transformation_data[tname].submodel = [name]
                #nest_level = self._nest_level(submodel)
                if submodel._fixed:
                    self._fixed_vardata[name] = [vardata for v in submodel._fixed for vardata in v.values()]
                    instance._transformation_data[tname].fixed = [ComponentUID(v) for v in self._fixed_vardata[name]]
                    self._submodel[name] = submodel
                else:
                    e = "Must specify 'fixed' or 'unfixed' options"
                    logger.error(e)
                    raise RuntimeError(e)
                self._preprocess(tname,submodel)
        return

    def _fix_all(self):
        """
        Fix the upper variables
        """
        for key,vardata in self._fixed_vardata.items():
            for v in vardata:
                if not v.fixed:
                    self._fixed_ids.add(id(v))
                    v.fixed = True

    def _unfix_all(self):
        """
        Unfix the upper variables
        """
        for key,vardata in self._fixed_vardata.items():
            for v in vardata:
                if id(v) in self._fixed_ids:
                    v.fixed = False
                    self._fixed_ids.remove(id(v))

