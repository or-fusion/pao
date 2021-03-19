import math
import pyutilib.th as unittest
from pao.lbp import *
from pao.lbp import examples
import pyomo.opt


solvers = pyomo.opt.check_available_solvers('glpk','cbc','ipopt')


class Test_bilevel_FA(unittest.TestCase):

    # TODO - test with either cbc or glpk

    def test_bard511(self):
        lbp = examples.bard511.create()
        lbp.check()

        opt = Solver('pao.lbp.FA')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 4))

    def test_bard511_list(self):
        lbp = examples.bard511_list.create()
        lbp.check()

        opt = Solver('pao.lbp.FA')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.lbp.FA')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        lmp, soln = linearize_bilinear_terms(qmp) 
        lmp.check()
        opt.solve(lmp)
        soln.copy(From=lmp, To=qmp)

        self.assertTrue(math.isclose(qmp.U.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[1], 0))

    def test_besancon27(self):
        lbp = examples.besancon27.create()
        lbp.check()

        opt = Solver('pao.lbp.FA')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 0))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 1))

    def test_getachew_ex1(self):
        lbp = examples.getachew_ex1.create()
        lbp.check()

        opt = Solver('pao.lbp.FA')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 8))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 6))

    def test_getachew_ex2(self):
        lbp = examples.getachew_ex2.create()
        lbp.check()

        opt = Solver('pao.lbp.FA')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 6))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 8))

    def test_pineda(self):
        lbp = examples.pineda.create()
        lbp.check()

        opt = Solver('pao.lbp.FA')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 2))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 100))


@unittest.skipIf('ipopt' not in solvers, "Ipopt solver is not available")
class Test_bilevel_REG(unittest.TestCase):

    def test_bard511(self):
        lbp = examples.bard511.create()
        lbp.check()

        opt = Solver('pao.lbp.REG')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_bard511_list(self):
        lbp = examples.bard511_list.create()
        lbp.check()

        opt = Solver('pao.lbp.REG')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.lbp.REG')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            # This problem has binary upper-level variables
            pass

    def test_besancon27(self):
        lbp = examples.besancon27.create()
        lbp.check()

        opt = Solver('pao.lbp.REG')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 1, abs_tol=1e-4))

    def test_getachew_ex1(self):
        lbp = examples.getachew_ex1.create()
        lbp.check()

        opt = Solver('pao.lbp.REG')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        lbp = examples.getachew_ex2.create()
        lbp.check()

        opt = Solver('pao.lbp.REG')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 8, abs_tol=1e-4))

    def test_pineda(self):
        lbp = examples.pineda.create()
        lbp.check()

        opt = Solver('pao.lbp.REG')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 100, abs_tol=1e-4))


@unittest.skipIf('cbc' not in solvers, "CBC solver is not available")
class Test_bilevel_PCCG(unittest.TestCase):

    solver = 'cbc'

    def test_bard511(self):
        lbp = examples.bard511.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_bard511_list(self):
        lbp = examples.bard511_list.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.lbp.PCCG')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        lmp, soln = linearize_bilinear_terms(qmp) 
        lmp.check()
        opt.solve(lmp)
        soln.copy(From=lmp, To=qmp)

        self.assertTrue(math.isclose(qmp.U.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[1], 0))

    def test_besancon27(self):
        lbp = examples.besancon27.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 1, abs_tol=1e-4))

    def test_getachew_ex1(self):
        lbp = examples.getachew_ex1.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        lbp = examples.getachew_ex2.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 8, abs_tol=1e-4))

    def test_pineda(self):
        lbp = examples.pineda.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 100, abs_tol=1e-4))

    def test_toyexample1(self):
        lbp = examples.toyexample1.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertEqual(lbp.U.x.values[0], 2)
        self.assertEqual(lbp.U.LL.x.values[0], 2)

    def test_toyexample2(self):
        lbp = examples.toyexample2.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertEqual(lbp.U.x.values[0], 8)
        self.assertEqual(lbp.U.LL.x.values[0], 6)

    def test_toyexample3(self):
        lbp = examples.toyexample3.create()
        lbp.check()

        opt = Solver('pao.lbp.PCCG')
        opt.solve(lbp, solver=self.solver)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 3, abs_tol=1e-4))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 0.5, abs_tol=1e-4))
        self.assertEqual(lbp.U.x.values[1], 8)
        self.assertEqual(lbp.U.LL.x.values[1], 0)


#class Test_bilevel_ld(unittest.TestCase):
class XTest_bilevel_ld(object):

    def test_besancon27(self):
        lbp = examples.besancon27.create()
        lbp.check()
        lbp.print()

        opt = Solver('pao.bilevel.ld')
        opt.solve(lbp)

        self.assertTrue(math.isclose(lbp.U.x.values[0], 0))
        self.assertTrue(math.isclose(lbp.U.LL.x.values[0], 1))

if __name__ == "__main__":
    unittest.main()
