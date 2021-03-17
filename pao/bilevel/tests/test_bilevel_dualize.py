#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Test transformations for bilevel linear programs
#

from os.path import abspath, dirname, join
from parameterized import parameterized
import pyutilib.th as unittest

from pyomo.environ import *
from pao.bilevel.components import SubModel
from pyomo.core import Block, TransformationFactory

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for bilevel reformulation tests
reformulation_model_names = ['bqp_example1','bqp_example2']
reformulation_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in reformulation_model_names]
reformulations = [join(current_dir, 'auxiliary','reformulation','{}_dual.txt'.format(i)) for i in reformulation_model_names]

#class TestBilevelDualize(unittest.TestCase):
class XTestBilevelDualize(object):
    """
    Testing for bilevel dualization that use the pao.duality.linear_dual transformation for the SubModel

    """
    show_output = True

    @classmethod
    def setUpClass(self): pass

    @classmethod
    def setUp(self): pass

    @classmethod
    def tearDown(self): pass

    @parameterized.expand(zip(reformulation_model_names, reformulation_models, reformulations))
    def test_reformulation(self, name, model, reformulation):
        """ Tests bilevel reformulation and checks whether the derivation is equivalent
        to the known solution in the reformulation/*.out file

        Parameters
        ----------
        name : `string`
        model: `string`
        reformulation: `string`

        """
        from importlib.machinery import SourceFileLoader
        namespace = SourceFileLoader(name,model).load_module()
        instance = namespace.pyomo_create_model()

        xfrm = TransformationFactory('pao.duality.linear_dual')
        for submodel in instance.component_objects(SubModel, descend_into=True):
            instance.reclassify_component_type(submodel, Block)
            dualmodel = xfrm._create_using(instance, block=submodel.name)
            break

        with open(join(aux_dir, name + '_linear_mpec.out'), 'w') as ofile:
            dualmodel.pprint(ostream=ofile)

        self.assertFileEqualsBaseline(join(aux_dir, name + '_linear_mpec.out'),
                                      reformulation, tolerance=1e-5)

if __name__ == "__main__":
    unittest.main()
