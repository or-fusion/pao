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
import math
from parameterized import parameterized
import pyutilib.th as unittest

import pyomo.opt
from pyomo.environ import *
import itertools
from pyomo.core import Objective
from pao.bilevel.components import SubModel
from pao.bilevel.plugins.collect import collect_bilevel_matrix_representation

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for bilevel reformulation tests
reformulation_model_names = ['bqp_example1','bqp_example2']
reformulation_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in reformulation_model_names]
reformulations = [join(current_dir, 'auxiliary','reformulation','{}.txt'.format(i)) for i in reformulation_model_names]


class TestBilevelMatrixRepn(unittest.TestCase):
    """
    Testing for bilevel matrix representation of models

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

        (A_c, c_vars, sense_c, b_coef_c, \
            A_b, b_vars, sense_b, b_coef_b, \
            A_i, i_vars, sense_i, b_coef_i, \
            A_f, fixed_vars, sense_f, b_coef_f) = collect_bilevel_matrix_representation(instance)

        print(A_c, c_vars, sense_c, b_coef_c, \
            A_b, b_vars, sense_b, b_coef_b, \
            A_i, i_vars, sense_i, b_coef_i, \
            A_f, fixed_vars, sense_f, b_coef_f)

        for submodel in instance.component_objects(SubModel):
            (A_c, c_vars, sense_c, b_coef_c, \
            A_b, b_vars, sense_b, b_coef_b, \
            A_i, i_vars, sense_i, b_coef_i, \
            A_f, fixed_vars, sense_f, b_coef_f) = collect_bilevel_matrix_representation(submodel)

        print(A_c, c_vars, sense_c, b_coef_c, \
            A_b, b_vars, sense_b, b_coef_b, \
            A_i, i_vars, sense_i, b_coef_i, \
            A_f, fixed_vars, sense_f, b_coef_f)

        with open(join(aux_dir, name + '_linear_mpec.out'), 'w') as ofile:
            instance.pprint(ostream=ofile)

        self.assertFileEqualsBaseline(join(aux_dir, name + '_linear_mpec.out'),
                                      reformulation, tolerance=1e-5)

if __name__ == "__main__":
    unittest.main()
