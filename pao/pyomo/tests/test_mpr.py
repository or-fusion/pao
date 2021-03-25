import math
import pyutilib.th as unittest
from pao.pyomo import *
from pao.pyomo import examples
import pyomo.opt


solvers = pyomo.opt.check_available_solvers('glpk','cbc','ipopt','gurobi')


class Test_solver(unittest.TestCase):

    @unittest.skipIf('cbc' not in solvers, "CBC solver is not available")
    def test_FA(self):
        M = examples.bard511.create()

        opt = Solver('pao.pyomo.FA', mip_solver=Solver('cbc'))
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 4))
        self.assertTrue(math.isclose(M.y.value, 4))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver is not available")
    def test_REG(self):
        M = examples.bard511.create()

        opt = Solver('pao.pyomo.REG', nlp_solver=Solver('ipopt'))
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.y.value, 4, abs_tol=1e-4))

    @unittest.skipIf('cbc' not in solvers, "CBC solver is not available")
    def test_PCCG(self):
        M = examples.bard511.create()

        opt = Solver('pao.pyomo.PCCG', mip_solver=Solver('cbc'))
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 4))
        self.assertTrue(math.isclose(M.y.value, 4))


class Test_pyomo_FA(unittest.TestCase):

    # TODO - test with either cbc or glpk

    def test_bard511(self):
        M = examples.bard511.create()

        opt = Solver('pao.pyomo.FA')
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 4))
        self.assertTrue(math.isclose(M.y.value, 4))

    def test_besancon27(self):
        M = examples.besancon27.create()

        opt = Solver('pao.pyomo.FA')
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 0))
        self.assertTrue(math.isclose(M.v.value, 1))

    def test_barguel(self):
        M = examples.barguel.create()

        opt = Solver('pao.pyomo.FA')
        try:
            opt.solve(M)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        opt.solve(M, linearize_bigm=1e6)

        self.assertTrue(math.isclose(M.u.value, 0))
        self.assertTrue(math.isclose(M.x.value, 0))
        self.assertTrue(math.isclose(M.y.value, 0))

    def test_getachew_ex1(self):
        M = examples.getachew_ex1.create()

        opt = Solver('pao.pyomo.FA')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 8))
        self.assertTrue(math.isclose(M.L.xR.value, 6))

    def test_getachew_ex2(self):
        M = examples.getachew_ex2.create()

        opt = Solver('pao.pyomo.FA')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 6))
        self.assertTrue(math.isclose(M.L.xR.value, 8))

    def test_pineda(self):
        M = examples.pineda.create()

        opt = Solver('pao.pyomo.FA')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 2))
        self.assertTrue(math.isclose(M.L.xR.value, 100))

    #@unittest.skipIf('gurobi' not in solvers, "Gurobi solver is not available")
    def Xtest_sip_example1(self):
        M = examples.sip_example1.create()

        opt = Solver('pao.pyomo.FA')
        opt.solve(M, solver='gurobi')

        self.assertTrue(False)


@unittest.skipIf('ipopt' not in solvers, "Ipopt solver is not available")
class Test_pyomo_REG(unittest.TestCase):

    def test_bard511(self):
        M = examples.bard511.create()

        opt = Solver('pao.pyomo.REG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.y.value, 4, abs_tol=1e-4))

    def test_besancon27(self):
        M = examples.besancon27.create()

        opt = Solver('pao.pyomo.REG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.v.value, 1, abs_tol=1e-4))

    def test_getachew_ex1(self):
        M = examples.getachew_ex1.create()

        opt = Solver('pao.pyomo.REG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        M = examples.getachew_ex2.create()

        opt = Solver('pao.pyomo.REG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 8, abs_tol=1e-4))

    def test_pineda(self):
        M = examples.pineda.create()

        opt = Solver('pao.pyomo.REG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 100, abs_tol=1e-4))


@unittest.skipIf('cbc' not in solvers, "CBC solver is not available")
class Test_pyomo_PCCG(unittest.TestCase):

    def test_bard511(self):
        M = examples.bard511.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.y.value, 4, abs_tol=1e-4))

    def test_besancon27(self):
        M = examples.besancon27.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.x.value, 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.v.value, 1, abs_tol=1e-4))

    def test_barguel(self):
        M = examples.barguel.create()

        opt = Solver('pao.pyomo.PCCG')
        try:
            opt.solve(M)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        opt.solve(M, linearize_bigm=1e6)

        self.assertTrue(math.isclose(M.u.value, 0))
        self.assertTrue(math.isclose(M.x.value, 0))
        self.assertTrue(math.isclose(M.y.value, 0))

    def test_getachew_ex1(self):
        M = examples.getachew_ex1.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        M = examples.getachew_ex2.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 8, abs_tol=1e-4))

    def test_pineda(self):
        M = examples.pineda.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 100, abs_tol=1e-4))

    def test_toyexample1(self):
        M = examples.toyexample1.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertEqual(M.xZ.value, 2)
        self.assertEqual(M.L.xZ.value, 2)

    def test_toyexample2(self):
        M = examples.toyexample2.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertEqual(M.xZ.value, 8)
        self.assertEqual(M.L.xZ.value, 6)

    def test_toyexample3(self):
        M = examples.toyexample3.create()

        opt = Solver('pao.pyomo.PCCG')
        opt.solve(M)

        self.assertTrue(math.isclose(M.xR.value, 3, abs_tol=1e-4))
        self.assertTrue(math.isclose(M.L.xR.value, 0.5, abs_tol=1e-4))
        self.assertEqual(M.xZ.value, 8)
        self.assertEqual(M.L.xZ.value, 0)


if __name__ == "__main__":
    unittest.main()
