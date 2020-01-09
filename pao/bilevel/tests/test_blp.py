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

import sys
import os
from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = currdir #normpath(join(currdir,'..','..','..','examples','bilevel'))

import pyutilib.th as unittest
import pyutilib.misc

import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
from pyomo.environ import *
import pao

from six import iteritems

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

solvers = pyomo.opt.check_available_solvers('cplex', 'glpk', 'ipopt')

class CommonTests:

    solve = True
    solver='pao.bilevel.blp_global'

    def run_bilevel(self, *_args, **kwds):
        if self.solve:
            args = ['solve']
            if 'solver' in kwds:
                _solver = kwds.get('solver','glpk')
                args.append('--solver='+self.solver)
                args.append('--solver-options="solver=%s"' % _solver)
            args.append('--save-results=result.yml')
            args.append('--results-format=yaml')
        else:
            args = ['convert']
        if 'preprocess' in kwds:
            pp = kwds['preprocess']
            if pp == 'linear_mpec':
                args.append('--transform=pao.bilevel.linear_mpec')
        args.append('-c')

        # These were being ignored by the solvers for this package,
        # which now causes a helpful error message
        #args.append('--symbolic-solver-labels')
        #args.append('--file-determinism=2')

        if False:
            args.append('--stream-solver')
            args.append('--tempdir='+currdir)
            args.append('--keepfiles')
            args.append('--logging=debug')

        args = args + list(_args)
        os.chdir(currdir)

        print('***')
        #print(' '.join(args))
        try:
            output = pyomo_main.main(args)
        except SystemExit:
            output = None
        except:
            output = None
            raise
        cleanup()
        print('***')
        return output

    def check(self, problem, solver):
        pass

    def referenceFile(self, problem, solver):
        return join(currdir, problem+'.txt')

    def getObjective(self, fname):
        FILE = open(fname,'r')
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

    @unittest.category('fragile')
    def test_bard511(self):
        self.problem='test_bard511'
        self.run_bilevel( join(exdir,'bard_5_1_1.py') )
        self.check( 'bard511', 'linear_mpec' )


class Reformulate(unittest.TestCase, CommonTests):

    solve = False

    @classmethod
    def setUpClass(cls):
        import pao

    def tearDown(self):
        if os.path.exists(os.path.join(currdir, 'result.yml')):
            os.remove(os.path.join(currdir, 'result.yml'))

    def run_bilevel(self,  *args, **kwds):
        module = pyutilib.misc.import_file(args[0])
        instance = module.pyomo_create_model(None, None)
        xfrm = TransformationFactory('pao.bilevel.linear_mpec')
        xfrm.apply_to(instance, deterministic=True)
        with open(join(currdir, self.problem+'_linear_mpec.out'), 'w') as ofile:
            instance.pprint(ostream=ofile)

    def referenceFile(self, problem, solver):
        return join(currdir, 'test_'+problem+"_linear_mpec.txt")

    def check(self, problem, solver):
        self.assertFileEqualsBaseline( join(currdir,self.problem+'_linear_mpec.out'),
                                           self.referenceFile(problem,solver), tolerance=1e-5 )

    #@unittest.category('fragile')
    def test_bqp1(self):
        self.problem='test_bqp1'
        self.run_bilevel( join(exdir,'bqp_example1.py') )
        self.check( 'bqp1', 'linear_mpec' )

    def test_bqp2(self):
        self.problem='test_bqp2'
        self.run_bilevel( join(exdir,'bqp_example2.py') )
        self.check( 'bqp2', 'linear_mpec' )


class Solver(unittest.TestCase):

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
                #self.assertEqual(val['Id'], ansObj[i].get(key,None)['Id'])
                self.assertAlmostEqual(val['Value'], ansObj[i].get(key,None)['Value'], places=3)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'glpk' in solvers, "The 'glpk' executable is not available")
class Solve_GLPK(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        import pao

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'glpk'
        CommonTests.run_bilevel(self, *args, **kwds)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'cplex' in solvers, "The 'cplex' executable is not available")
class Solve_CPLEX(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        import pao

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'cplex'
        CommonTests.run_bilevel(self, *args, **kwds)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'ipopt' in solvers, "The 'ipopt' executable is not available")
class Solve_IPOPT(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        import pao

    solver='pao.bilevel.blp_local'

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'ipopt'
        CommonTests.run_bilevel(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
