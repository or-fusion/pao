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

solvers = pyomo.opt.check_available_solvers('cplex','gurobi')
pao_solvers = ['pao.bilevel.ccg']

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for integer bilevel solution tests
solution_model_names = ['yueA1','yueA2','yueA3']
solution_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in solution_model_names]
solutions = [-22, -20, -243.500]

# cartesian product of lists for a full coverage unittest run
cartesian_solutions = [elem for elem in itertools.product(*[solvers,pao_solvers,zip(solution_model_names,solution_models,solutions)])]

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
        results = solver.solve(instance, tee=False)

        self.assertTrue(results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal)

        test_objective = self.getObjectiveInstance(instance)
        comparison = math.isclose(test_objective,solution,rel_tol=1e-3)
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

    def getObjectiveInstance(self, instance):
        """ Gets the master problem objective solution from the unittest instance

        Parameters
        ----------
        instance : `string`
        ans: `float`

        Returns
        -------
        `float`
        """

        for (name, data) in instance.component_map(active=True).items():
            if isinstance(data, Objective):
                return value(data)



if __name__ == "__main__":
    unittest.main()
