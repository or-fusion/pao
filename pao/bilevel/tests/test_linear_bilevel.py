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

import os
from os.path import abspath, dirname, join
import math
#import unittest
from parameterized import parameterized
import pyutilib.th as unittest
import pyutilib.misc

import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
from pyomo.environ import *
import itertools

from six import iteritems

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

solvers = pyomo.opt.check_available_solvers('cplex','glpk','gurobi')
#pao_solvers = ['pao.bilevel.ld','pao.bilevel.blp_global','pao.bilevel.blp_local','pao.bilevel.bqp']
pao_solvers = ['pao.bilevel.blp_global']

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'aux')

reformulation_model_names = ['bqp_example1','bqp_example2']
reformulation_models = [join(current_dir, 'aux', '{}.py'.format(i)) for i in reformulation_model_names]
reformulations = [join(current_dir, 'aux','reformulation','{}.txt'.format(i)) for i in reformulation_model_names]

solution_model_names = ['bard511']
solution_models = [join(current_dir, 'aux', '{}.py'.format(i)) for i in solution_model_names]
solutions = [join(current_dir, 'aux','solution','{}.txt'.format(i)) for i in solution_model_names]

cartesian_solutions = [elem for elem in itertools.product(*[solvers,pao_solvers,zip(solution_model_names,solution_models,solutions)])]


class TestBilevelReformulate(unittest.TestCase):
    show_output = True

    @classmethod
    def setUpClass(self): pass

    @classmethod
    def setUp(self): pass

    @classmethod
    def tearDown(self): pass

    @parameterized.expand(zip(reformulation_model_names, reformulation_models, reformulations))
    def test_reformulation(self, name, model, reformulation):
        from importlib.machinery import SourceFileLoader
        namespace = SourceFileLoader(name,model).load_module()
        instance = namespace.pyomo_create_model()

        xfrm = TransformationFactory('pao.bilevel.linear_mpec')
        xfrm.apply_to(instance, deterministic=True)

        with open(join(aux_dir, name + '_linear_mpec.out'), 'w') as ofile:
            instance.pprint(ostream=ofile)

        self.assertFileEqualsBaseline(join(aux_dir, name + '_linear_mpec.out'),
                                      reformulation, tolerance=1e-5)


class TestBilevelSolve(unittest.TestCase):
    show_output = True

    @classmethod
    def setUpClass(self): pass

    @classmethod
    def setUp(self): pass

    @classmethod
    def tearDown(self): pass

    @parameterized.expand(cartesian_solutions)
    def test_solution(self, solver, pao_solver, solution_zip):
        (name, model, solution) = solution_zip
        from importlib.machinery import SourceFileLoader
        namespace = SourceFileLoader(name,model).load_module()
        instance = namespace.pyomo_create_model()

        solver = SolverFactory(solver)
        solver.set_options(('"solver=%s"' % pao_solver))
        results = solver.solve(instance)

        solution_objective = self.getObjective(solution)

        self.assertTrue(results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal)

        self.assertAlmostEqual(value(instance.o.expr), solution_objective, places=3)

    def getObjective(self, filename):
        FILE = open(filename,'r')
        data = yaml.load(FILE, Loader=yaml.SafeLoader)
        FILE.close()
        solutions = data.get('Solution', [])
        ans = []
        for x in solutions:
            ans.append(x.get('Objective', {}))
        return ans

if __name__ == "__main__":
    unittest.main()
