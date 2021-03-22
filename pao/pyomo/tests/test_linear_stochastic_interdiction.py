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
from pao.pyomo.components import SubModel

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

# TODO: Add glpk in solvers list

solvers = pyomo.opt.check_available_solvers('cplex','gurobi')
pao_solvers = ['pao.bilevel.stochastic_ld']

current_dir = dirname(abspath(__file__))
aux_dir = join(dirname(abspath(__file__)),'auxiliary')

# models for bilevel solution tests
solution_model_names = ['sip_example1']
solution_models = [join(current_dir, 'auxiliary', '{}.py'.format(i)) for i in solution_model_names]
solutions = [join(current_dir, 'auxiliary','solution','{}.txt'.format(i)) for i in solution_model_names]

# cartesian product of lists for a full coverage unittest run
cartesian_solutions = [elem for elem in itertools.product(*[solvers,pao_solvers,zip(solution_model_names,solution_models,solutions)])]
cartesian_solutions = []


@unittest.skipIf(len(cartesian_solutions)==0, "No solvers available")
class TestStochasticBilevelSolve(unittest.TestCase):
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

    @parameterized.expand(cartesian_solutions, skip_on_empty=True)
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
        weights = instance.weights
        kwargs = {'subproblem_objective_weights': weights}
        solver = SolverFactory(pao_solver)
        solver.options.solver = numerical_solver
        results = solver.solve(instance, **kwargs, tee=False)

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
