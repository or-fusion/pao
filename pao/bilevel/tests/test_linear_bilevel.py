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

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

# only runs with no error when solvers = ['ipopt'] and pao_solvers = ['pao.bilevel.blp_local']
#solvers = pyomo.opt.check_available_solvers('cplex','glpk','gurobi','ipopt')
solvers = ['ipopt']
pao_solvers = ['pao.bilevel.blp_local']#,'pao.bilevel.blp_global']
solvers2 = pyomo.opt.check_available_solvers('cplex','glpk','gurobi','ipopt')
pao_solvers2 = ['pao.bilevel.ld']

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for bilevel reformulation tests
reformulation_model_names = ['bqp_example1','bqp_example2']
reformulation_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in reformulation_model_names]
reformulations = [join(current_dir, 'auxiliary','reformulation','{}.txt'.format(i)) for i in reformulation_model_names]

# models for bilevel solution tests
solution_model_names = ['bard511']
solution_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in solution_model_names]
solutions = [join(current_dir, 'auxiliary','solution','{}.txt'.format(i)) for i in solution_model_names]

solution_model_names2 = ['t5','t1','t1b']
solution_models2 = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in solution_model_names2]
solutions2 = [join(current_dir, 'auxiliary','solution','{}.txt'.format(i)) for i in solution_model_names2]

# cartesian product of lists for a full coverage unittest run
cartesian_solutions = [elem for elem in itertools.product(*[solvers,pao_solvers,zip(solution_model_names,solution_models,solutions)])]
cartesian_solutions2 = [elem for elem in itertools.product(*[solvers2,pao_solvers2,zip(solution_model_names2,solution_models2,solutions2)])]
cartesian_solutions = cartesian_solutions + cartesian_solutions2

class TestBilevelReformulate(unittest.TestCase):
    """
    Testing for bilevel reformulations that use the pao.bilevel.linear_mpec transformation

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

        xfrm = TransformationFactory('pao.bilevel.linear_mpec')
        xfrm.apply_to(instance, deterministic=True)

        with open(join(aux_dir, name + '_linear_mpec.out'), 'w') as ofile:
            instance.pprint(ostream=ofile)

        self.assertFileEqualsBaseline(join(aux_dir, name + '_linear_mpec.out'),
                                      reformulation, tolerance=1e-5)


class TestBilevelSolve(unittest.TestCase):
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
    def test_solution(self, numerical_solver, pao_solver, solution_zip):
        """ Tests bilevel solution and checks whether the derivation is equivalent
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

        solver = SolverFactory(pao_solver)
        solver.options.solver = numerical_solver
        results = solver.solve(instance)

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
            if isinstance(data, SubModel):
                root_name = name
                self.getObjectiveInstance(data, root_name, ans)
        return ans



if __name__ == "__main__":
    unittest.main()
