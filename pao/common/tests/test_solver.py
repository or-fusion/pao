import math
import pyutilib.th as unittest
import pyomo.environ as pe

from pao import Solver
import pyomo.opt
from pyomo.neos.kestrel import kestrelAMPL

neos_available = False
try:
    raise RuntimeError("Disable NEOS Tests")
    if kestrelAMPL().neos is not None:
        neos_available = True
except:
    pass

solvers = pyomo.opt.check_available_solvers('glpk','ipopt')


def create_lp1():
    M = pe.ConcreteModel()
    M.x = pe.Var(bounds=(0,None))
    M.y = pe.Var(bounds=(0,None))
    M.o = pe.Objective(expr=2*M.x+M.y, sense=pe.maximize)
    M.c = pe.Constraint(expr=M.x+M.y == 1)
    return M

def create_nlp1():
    M = pe.ConcreteModel()
    A = list(range(10))
    M.x = pe.Var(A, bounds=(0,None), initialize=1)
    M.o = pe.Objective(expr=sum(pe.sin((i+1)*M.x[i]) for i in A))
    M.c = pe.Constraint(expr=sum(M.x[i] for i in A) >= 1)
    return M


@unittest.skipIf('ipopt' not in solvers, "Ipopt solver is not available")
class Test_ipopt(unittest.TestCase):

    def test_lp1(self):
        M = create_lp1()
        opt = Solver('ipopt')
        self.assertTrue(opt.available())
        res = opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))

    def test_nlp1(self):
        M = create_nlp1()
        opt = Solver('ipopt', tee=True, max_cpu_time=1e-12, print_level=0)
        self.assertTrue(opt.available())
        #opt.config.display()
        #print(opt.solver.options)
        res = opt.solve(M)
        res = opt.solve(M, max_cpu_time=100)
        # Solving with a solver option doesn't change the value configured when creating the solver
        self.assertTrue(math.isclose(opt.solver.options.max_cpu_time, 1e-12))

@unittest.skipIf('glpk' not in solvers, "GLPK solver is not available")
class Test_glpk(unittest.TestCase):

    def test_lp1(self):
        M = create_lp1()
        opt = Solver('glpk')
        self.assertTrue(opt.available())
        res = opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))


class Test_foobar(unittest.TestCase):

    def test_lp1(self):
        try:
            Solver('foobar')
        except AssertionError:
            pass

    def test_glpk_executable(self):
        try:
            opt = Solver('glpk', executable="glpk_foobar")
        except ValueError:
            pass
        #
        opt = Solver('glpk')
        opt.config.executable = 'glpk_foobar'
        try:
            self.assertTrue(opt.available())
        except ValueError:
            pass
        #
        M = create_lp1()
        opt = Solver('glpk')
        opt.config.executable = 'glpk_foobar'
        try:
            self.assertTrue(opt.solve(M))
        except ValueError:
            pass



class Test_neos(unittest.TestCase):

    @unittest.skipIf(not neos_available, "NEOS not available")
    def test_neos_ipopt_available(self):
        M = create_nlp1()
        opt = Solver('ipopt', server='neos', email='pao@gmail.com', max_cpu_time=1e-12)
        self.assertTrue(opt.available())
        res = opt.solve(M)
        #M.x.pprint()
        res = opt.solve(M, max_cpu_time=100)
        self.assertTrue(math.isclose(opt.solver_options['max_cpu_time'], 1e-12))
        #print(res)
        #M.x.pprint()

    @unittest.skipIf(not neos_available, "NEOS not available")
    def test_neos_cbc_available(self):
        M = create_lp1()
        opt = Solver('cbc', server='neos', email='pao@gmail.com')
        self.assertTrue(opt.available())
        res = opt.solve(M, tee=True)
        
        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))

    @unittest.skipIf(not neos_available, "NEOS not available")
    def test_neos_cbc_options(self):
        M = create_lp1()
        opt = Solver('cbc', server='neos', email='pao@gmail.com', foo=1, bar=None)
        self.assertTrue(opt.available())
        res = opt.solve(M)
        
        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))

    # This is slow
    @unittest.skipIf(neos_available, "NEOS is available")
    def Xtest_neos_cbc_missing(self):
        try:
            opt = Solver('cbc', server='neos')
        except AssertionError:
            pass

    @unittest.skipIf(not neos_available, "NEOS not available")
    def test_neos_foobar(self):
        M = create_lp1()
        opt = Solver('foobar', server='neos')
        try:
            res = opt.solve(M)
        except RuntimeError:
            pass


if __name__ == "__main__":
    unittest.main()
