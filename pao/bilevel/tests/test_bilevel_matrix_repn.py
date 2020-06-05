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
from pao.bilevel.plugins.collect import BilevelMatrixRepn

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for bilevel reformulation tests
reformulation_model_names = ['yueA3']
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

        matrix_repn = BilevelMatrixRepn(instance)
        print('---------------------')
        print('--Variable Vector----')
        print('---------------------')
        _name = [var.name for idx,var in matrix_repn._all_vars.items()]
        print(_name)
        print('\n')
        print('---------------------')
        print('-Grouped By Variable-')
        print('---------------------')
        for submodel in instance.component_objects(SubModel):
            for var in instance.component_objects(Var):
                (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var)
                (C, C_q, C_constant) = matrix_repn.cost_vectors(submodel, var)
                print('---------------------')
                print('submodel name: {}'.format(submodel.name))
                print('variable name: {}'.format(var.name))
                print('A linear matrix coefficients: ')
                print(A)
                print('A bilinear matrix coefficients: ')
                print(A_q.toarray())
                print('Sense (=, <=, >=): ')
                print(sign)
                print('Rhs: ')
                print(b)
                print('C linear coefficient: ')
                print(C)
                print('C bilinear vector coefficients: ')
                print(C_q)
                print('C constant coefficient: ')
                print(C_constant)
        print('\n')
        print('---------------------')
        print('--Grouped By Sense---')
        print('---------------------')
        for submodel in instance.component_objects(SubModel):
            for var in instance.component_objects(Var):
                for sense in ['e', 'l']:
                    (A, A_q, sign, b) = matrix_repn.coef_matrices(submodel, var, sense=sense)
                    print('---------------------')
                    print('submodel name: {}'.format(submodel.name))
                    print('variable name: {}'.format(var.name))
                    print('A linear matrix coefficients: ')
                    print(A)
                    print('A bilinear matrix coefficients: ')
                    print(A_q.toarray())
                    print('Sense (=, <=, >=): ')
                    print(sign)
                    print('Rhs: ')
                    print(b)

        from pao.bilevel.solvers.solver6 import BilevelSolver6
        solver = BilevelSolver6()
        solver._presolve(instance)
        solver._apply_solver
        solver._postsolve
        # solver = SolverFactory('pao.bilevel.ccg')
        # solver.options.solver = 'gurobi'
        # results = solver.solve(instance, tee=False)

if __name__ == "__main__":
    unittest.main()
