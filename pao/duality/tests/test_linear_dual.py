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
# Test transformations for linear duality
#

import os
from os.path import abspath, dirname, normpath, join
from six import iteritems
try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

import pyutilib.misc
from pyutilib.misc import Options, Container
import pyutilib.th as unittest

from pyomo.environ import TransformationFactory, SolverFactory, ComponentUID
import pyomo.opt
import pao

currdir = dirname(abspath(__file__))
exdir = currdir
solver = None



class CommonTests(object):

    solve = True

    def run_bilevel(self, *_args, **kwds):
        os.chdir(currdir)

        try:
            #
            # Import the model file to create the model
            #
            usermodel = pyutilib.misc.import_file(_args[0], clear_cache=True)
            instance = usermodel.model
            #
            # Collected fixed variables
            #
            _fixed = kwds.pop('fixed', [])
            for v in _fixed:
                v_ = ComponentUID(v).find_component_on(instance)
            fixed = [ComponentUID(v).find_component_on(instance) for v in _fixed]
            #
            # Apply transformations
            #
            if 'transform' in kwds:
                xfrm = TransformationFactory(kwds['transform'])
                transform_kwds = kwds.get('transform_kwds', {})
                transform_kwds['fixed'] = fixed
                new_instance = xfrm.create_using(instance, **transform_kwds)
            else:
                new_instance = instance

            if self.solve:
                #
                # Solve the problem
                #
                opt = SolverFactory( kwds.get('solver', 'glpk') )
                results = opt.solve(new_instance)
                new_instance.solutions.store_to(results)
                with open('result.yml', 'w') as ofile:
                    results.write(ostream=ofile, format='json')
            elif kwds.get('format', 'lp') == 'lp':
                #
                # Write the file
                #
                io_options = {}
                io_options['symbolic_solver_labels'] = True
                io_options['file_determinism'] = 2
                new_instance.name = 'Test'
                new_instance.write(self.problem+"_result.lp", io_options=io_options)
            else:
                #
                # This is a hack.  When we cannot write a valid LP file, we still
                # write with the LP suffix to simplify the testing logic.
                #
                with open(self.problem+"_result.lp", 'w') as ofile:
                    new_instance.pprint(ostream=ofile)

        except:
            print("Failed to construct and transform the model")
            raise

    def check(self, problem, solver):
        pass

    def referenceFile(self, problem, solver):
        return join(currdir, problem+'.txt')

    def getObjective(self, fname):
        FILE = open(fname)
        data = yaml.load(FILE, Loader=yaml.SafeLoader)
        FILE.close()
        solutions = data.get('Solution', [])
        ans = []
        for x in solutions:
            ans.append(x.get('Objective', {}))
        return ans

    def updateDocStrings(self):
        for key in dir(self):
            if key.startswith('test'):
                getattr(self,key).__doc__ = " (%s)" % getattr(self,key).__name__

    def test_t1(self):
        self.problem='test_t1'
        self.run_bilevel(join(exdir,'t1.py'))
        self.check( 't1', 'linear_dual' )

    def test_t2(self):
        self.problem='test_t2'
        self.run_bilevel(join(exdir,'t2.py'))
        self.check( 't2', 'linear_dual' )

    def test_t3(self):
        self.problem='test_t3'
        self.run_bilevel(join(exdir,'t3.py'))
        self.check( 't3', 'linear_dual' )

    def test_t5(self):
        self.problem='test_t5'
        self.run_bilevel(join(exdir,'t5.py'))
        self.check( 't5', 'linear_dual' )

    def test_t6(self):
        self.problem='test_t6'
        self.run_bilevel(join(exdir,'t6.py'))
        self.check( 't6', 'linear_dual' )


class Reformulate(unittest.TestCase, CommonTests):

    solve=False

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    @classmethod
    def setUpClass(cls):
        import pao

    def run_bilevel(self,  *args, **kwds):
        args = list(args)
        args.append('--output='+self.problem+'_result.lp')
        kwds['transform'] = 'pao.duality.linear_dual'
        CommonTests.run_bilevel(self, *args, **kwds)

    def referenceFile(self, problem, solver):
        return join(currdir, problem+"_"+solver+'.lp')

    def check(self, problem, solver):
        resfile = join(currdir,self.problem+'_result.lp')
        self.assertFileEqualsBaseline( resfile, self.referenceFile(problem,solver), tolerance=1e-5 )

    def test_t1a(self):
        self.problem='test_t1a'
        self.run_bilevel(join(exdir,'t1a.py'), transform_kwds={'block':'b'})
        self.check( 't1', 'linear_dual' )

    def test_t1b(self):
        self.problem='test_t1b'
        self.run_bilevel(join(exdir,'t1a.py'), transform_kwds={'block':'B'})
        self.check( 't1b', 'linear_dual' )

    def test_t2a(self):
        self.problem='test_t2a'
        self.run_bilevel(join(exdir,'t2a.py'), transform_kwds={'block':'b'})
        self.check( 't2', 'linear_dual' )

    def test_t2b(self):
        self.problem='test_t2b'
        self.run_bilevel(join(exdir,'t2a.py'), transform_kwds={'block':'B'})
        self.check( 't2b', 'linear_dual' )

    def test_t2c(self):
        self.problem='test_t2c'
        try:
            self.run_bilevel(join(exdir,'t2a.py'), transform_kwds={'block':'C'})
            self.fail("Expected RuntimeError because of missing block")
        except RuntimeError:
            pass

    def test_t3_fixedsome(self):
        self.problem='test_t3_fixedsome'
        self.run_bilevel(join(exdir,'t3.py'), fixed=['x2','b.x1'], format='txt')
        self.check( 't3_fixedsome', 'linear_dual' )

    def test_t10(self):
        self.problem='test_t10'
        self.run_bilevel(join(exdir,'t10.py'))
        self.check( 't10', 'linear_dual' )

    def test_t11(self):
        self.problem='test_t11'
        self.run_bilevel(join(exdir,'t11.py'))
        self.check( 't11', 'linear_dual' )

    def test_err1(self):
        self.problem='test_err1'
        try:
            self.run_bilevel(join(exdir,'err1.py'))
            self.fail("Expected RuntimeError because model contains multiple objective expressions.")
        except RuntimeError:
            pass

    def test_err2(self):
        self.problem='test_err2'
        try:
            self.run_bilevel(join(exdir,'err2.py'))
            self.fail("Expected RuntimeError because model contains no objective expression.")
        except RuntimeError:
            pass

    def test_t3_fixedall(self):
        self.problem='test_t3_fixedall'
        try:
            self.run_bilevel(join(exdir,'t3.py'), fixed=['x1','x2','b.x1'])
            self.fail("Expected RuntimeError because model contains no objective expression.")
        except RuntimeError:
            pass



class Solver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pao

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    def check(self, problem, solver):
        refObj = self.getObjective(self.referenceFile(problem,solver))
        ansObj = self.getObjective(join(currdir,'result.yml'))
        self.assertEqual(len(refObj), len(ansObj))
        for i in range(len(refObj)):
            self.assertEqual(len(refObj[i]), len(ansObj[i]))
            for key,val in iteritems(refObj[i]):
                self.assertAlmostEqual(val['Value'], ansObj[i].get(key,None)['Value'], places=3)


class Solve_GLPK(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pao
        solvers = pyomo.opt.check_available_solvers('glpk')

    def setUp(self):
        if not yaml_available:
            self.skipTest("YAML is not available")
        if not 'glpk' in solvers:
            self.skipTest("The 'glpk' executable is not available")

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'glpk'
        CommonTests.run_bilevel(self, *args, **kwds)


class Solve_CPLEX(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pao
        solvers = pyomo.opt.check_available_solvers('cplex')

    def setUp(self):
        if not yaml_available:
            self.skipTest("YAML is not available")
        if not 'cplex' in solvers:
            self.skipTest("The 'cplex' executable is not available")

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'cplex'
        CommonTests.run_bilevel(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
