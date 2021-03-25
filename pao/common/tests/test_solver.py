import math
import pyutilib.th as unittest
import pyomo.environ as pe

from pao import Solver
import pyomo.opt
from pyomo.neos.kestrel import kestrelAMPL

neos_available = False
try:
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


@unittest.skipIf('ipopt' not in solvers, "Ipopt solver is not available")
class Test_ipopt(unittest.TestCase):

    def test_lp1(self):
        M = create_lp1()
        opt = Solver('ipopt')
        self.assertTrue(opt.available())
        res = opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))


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
        opt = Solver('glpk', executable="glpk_foobar")
        self.assertTrue(opt.available())


class Test_neos(unittest.TestCase):

    @unittest.skipIf(not neos_available, "NEOS not available")
    def test_neos_cbc_available(self):
        M = create_lp1()
        opt = Solver('cbc', server='neos', email='pao@gmail.com')
        self.assertTrue(opt.available())
        res = opt.solve(M)
        
        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))

    @unittest.skipIf(not neos_available, "NEOS not available")
    def Xtest_neos_cbc_options(self):
        M = create_lp1()
        opt = Solver('cbc', server='neos', email='pao@gmail.com', foo=1, bar=None)
        self.assertTrue(opt.available())
        res = opt.solve(M)
        
        self.assertTrue(math.isclose(M.x.value, 1, abs_tol=1e-6))
        self.assertTrue(math.isclose(M.y.value, 0, abs_tol=1e-6))

    @unittest.skipIf(neos_available, "NEOS is available")
    def test_neos_cbc_missing(self):
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
