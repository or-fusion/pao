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

from pyomo.core import Block
import math
import pyomo.opt
from os.path import abspath, dirname, join
from parameterized import parameterized
import pyutilib.th as unittest
import itertools
from pyomo.environ import *

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for bilevel highpoint relaxation tests
reformulation_model_names = ['besancon27']
reformulation_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in reformulation_model_names]
reformulations = [join(current_dir, 'auxiliary','reformulation','{}.txt'.format(i)) for i in reformulation_model_names]

solvers = pyomo.opt.check_available_solvers('cplex','glpk','gurobi','ipopt')

# models for bilevel solution tests
solution_model_names = ['besancon27']
solution_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in solution_model_names]
solutions = [join(current_dir, 'auxiliary','solution','{}.txt'.format(i)) for i in solution_model_names]

cartesian_solutions = [elem for elem in itertools.product(*[solvers,zip(solution_model_names,solution_models,solutions)])]

class TestBilevelHighpoint(unittest.TestCase):
    """
    Testing for bilevel highpoint relaxation that use the pao.bilevel.highpoint transformation

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

        xfrm = TransformationFactory('pao.bilevel.highpoint')
        xfrm.apply_to(instance, deterministic=True)

        with open(join(aux_dir, name + '_highpoint.out'), 'w') as ofile:
            instance.pprint(ostream=ofile)

        self.assertFileEqualsBaseline(join(aux_dir, name + '_highpoint.out'),
                                      reformulation, tolerance=1e-5)


class TestHighpointSolve(unittest.TestCase):
    """
    Testing for bilevel solutions that use the runtime parameters specified in cartesian_solutions list

    """
    show_output = True

    @classmethod
    def setUpClass(self): pass

    @classmethod
    def setUp(self): pass

    @classmethod
    def tearDown(self): pass

    @parameterized.expand(cartesian_solutions)
    def test_solution(self, numerical_solver, solution_zip):
        """ Tests highpoint relaxation solution and checks whether the derivation is equivalent
        to the known solution in the solution/*.txt file by checking for optimality and
        then comparing the value of the objective in the upper-level and all lower-levels

        Parameters
        ----------
        numerical_solver : `string`
        pao_solver: `string`
        solution_zip: tuple of three parameters (all of type `string`)

        """
        (name, model, solution) = solution_zip
        from importlib.machinery import SourceFileLoader
        namespace = SourceFileLoader(name,model).load_module()
        instance = namespace.pyomo_create_model()

        xfrm = TransformationFactory('pao.bilevel.highpoint')
        xfrm.apply_to(instance, deterministic=True)

        solver = SolverFactory(numerical_solver)
        for c in instance.component_objects(Block, descend_into=False):
            if 'hpr' in c.name:
                c.activate()
                results = solver.solve(c, tee=True, keepfiles=True)
                c.deactivate()

        self.assertTrue(results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal)

        test_objective = self.getObjectiveInstance(instance)
        solution_objective = self.getObjectiveSolution(solution, test_objective.keys())
        for key,val in test_objective.items():
            if key in solution_objective.keys():
                solution_val = solution_objective[key]
                comparison = math.isclose(val,solution_val,rel_tol=1e-3)
                self.assertTrue(comparison)

    def getObjectiveSolution(self, filename, keys):
        """ Gets the objective solutions from the known solution file that maps to
        the objective keys from the unittest run

        Parameters
        ----------
        filename : `string`
        keys: `dict_keys`

        Returns
        -------
        `dict`
        """

        FILE = open(filename,'r')
        data = yaml.load(FILE, Loader=yaml.SafeLoader)
        FILE.close()
        solutions = data.get('Solution', [])
        ans = dict()
        for x in solutions:
            tmp = x.get('Objective', {})
            if tmp != {}:
                for key in keys:
                    if key in tmp.keys():
                        val = tmp.get(key).get('Value')
                        ans[key] = val
        return ans

    def getObjectiveInstance(self, instance, root_name=None, ans=dict()):
        """ Gets the objective solutions from the unittest instance

        Parameters
        ----------
        instance : `string`
        root_name: `string`
        ans: `dict`

        Returns
        -------
        `dict`
        """

        for (name, data) in instance.component_map(active=True).items():
            if isinstance(data, Objective):
                if not root_name is None:
                    name = ("%s.%s" % (root_name, name))
                ans[name] = value(data)
        return ans


if __name__ == "__main__":
    unittest.main()
