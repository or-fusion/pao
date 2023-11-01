import math
import pyutilib.th as unittest
from pao.mpr import *
from pao.mpr import examples
import pyomo.opt


solvers = pyomo.opt.check_available_solvers('glpk','cbc','ipopt','mibs')


class Test_bilevel_FA(unittest.TestCase):

    # TODO - test with either cbc or glpk

    def test_bard511(self):
        mpr = examples.bard511.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4))

    def test_bard511_list(self):
        mpr = examples.bard511_list.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.mpr.FA')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        lmp, soln = linearize_bilinear_terms(qmp, 1e6)
        lmp.check()
        opt.solve(lmp)
        soln.copy(From=lmp, To=qmp)

        self.assertTrue(math.isclose(qmp.U.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[1], 0))

    def test_besancon27(self):
        mpr = examples.besancon27.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 0))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1))

    def test_besancon27_shifted(self):
        mpr = examples.besancon27_shifted.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2.5))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1.25))

    def test_getachew_ex1(self):
        mpr = examples.getachew_ex1.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 8))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 6))

    def test_getachew_ex2(self):
        mpr = examples.getachew_ex2.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 6))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 8))

    def test_pineda(self):
        mpr = examples.pineda.create()
        mpr.check()

        opt = Solver('pao.mpr.FA')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 100))


@unittest.skipIf('ipopt' not in solvers, "Ipopt solver is not available")
class Test_bilevel_REG(unittest.TestCase):

    def test_bard511(self):
        mpr = examples.bard511.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_bard511_list(self):
        mpr = examples.bard511_list.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.mpr.REG')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            # This problem has binary upper-level variables
            pass

    def test_besancon27(self):
        mpr = examples.besancon27.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1, abs_tol=1e-4))

    def test_besancon27_shifted(self):
        mpr = examples.besancon27_shifted.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2.5, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1.25, abs_tol=1e-4))

    def test_getachew_ex1(self):
        mpr = examples.getachew_ex1.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        mpr = examples.getachew_ex2.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 8, abs_tol=1e-4))

    def test_pineda(self):
        mpr = examples.pineda.create()
        mpr.check()

        opt = Solver('pao.mpr.REG')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 100, abs_tol=1e-4))


@unittest.skipIf('cbc' not in solvers, "CBC solver is not available")
class Test_bilevel_PCCG(unittest.TestCase):

    solver = 'cbc'

    def test_bard511(self):
        mpr = examples.bard511.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_bard511_list(self):
        mpr = examples.bard511_list.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.mpr.PCCG')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        lmp, soln = linearize_bilinear_terms(qmp, 1e6)
        lmp.check()
        opt.solve(lmp)
        soln.copy(From=lmp, To=qmp)

        self.assertTrue(math.isclose(qmp.U.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[1], 0))

    def test_besancon27(self):
        mpr = examples.besancon27.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1, abs_tol=1e-4))

    def test_besancon27_shifted(self):
        mpr = examples.besancon27_shifted.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2.5, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1.25, abs_tol=1e-4))

    def test_getachew_ex1(self):
        mpr = examples.getachew_ex1.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        mpr = examples.getachew_ex2.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 8, abs_tol=1e-4))

    def test_pineda(self):
        mpr = examples.pineda.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 100, abs_tol=1e-4))

    def test_toyexample1(self):
        mpr = examples.toyexample1.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertEqual(mpr.U.x.values[0], 2)
        self.assertEqual(mpr.U.LL.x.values[0], 2)

    def test_toyexample2(self):
        mpr = examples.toyexample2.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertEqual(mpr.U.x.values[0], 8)
        self.assertEqual(mpr.U.LL.x.values[0], 6)

    def test_toyexample3(self):
        mpr = examples.toyexample3.create()
        mpr.check()

        opt = Solver('pao.mpr.PCCG')
        opt.solve(mpr, mip_solver=self.solver)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 3, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 0.5, abs_tol=1e-4))
        self.assertEqual(mpr.U.x.values[1], 8)
        self.assertEqual(mpr.U.LL.x.values[1], 0)


@unittest.skipIf('mibs' not in solvers, "MibS solver is not available")
class Test_bilevel_MIBS(unittest.TestCase):

    def test_bard511(self):
        mpr = examples.bard511.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 4, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_bard511_list(self):
        mpr = examples.bard511_list.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 4, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 4, abs_tol=1e-4))

    def test_barguel(self):
        qmp = examples.barguel.create()
        qmp.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(qmp)
            self.fail("Expected an assertion error")
        except AssertionError:
            pass

        lmp, soln = linearize_bilinear_terms(qmp, 1e6)
        lmp.check()
        opt.solve(lmp)
        soln.copy(From=lmp, To=qmp)

        self.assertTrue(math.isclose(qmp.U.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[0], 0))
        self.assertTrue(math.isclose(qmp.U.LL.x.values[1], 0))

    def test_besancon27(self):
        mpr = examples.besancon27.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 0, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1, abs_tol=1e-4))

    def test_besancon27_shifted(self):
        mpr = examples.besancon27_shifted.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 2.5, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1.25, abs_tol=1e-4))

    def test_getachew_ex1(self):
        mpr = examples.getachew_ex1.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 8, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        mpr = examples.getachew_ex2.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 6, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 8, abs_tol=1e-4))

    def test_mibs(self):
        mpr = examples.mibs.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 6, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 5, abs_tol=1e-4))

    def test_moore(self):
        mpr = examples.moore.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 2, abs_tol=1e-4))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 2, abs_tol=1e-4))

    def test_pineda(self):
        mpr = examples.pineda.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 2, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 100, abs_tol=1e-4))

    def test_toyexample1(self):
        mpr = examples.toyexample1.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        opt.solve(mpr)

        self.assertEqual(mpr.U.x.values[0], 2)
        self.assertEqual(mpr.U.LL.x.values[0], 2)

    def test_toyexample2(self):
        mpr = examples.toyexample2.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')
        opt.solve(mpr)

        self.assertEqual(mpr.U.x.values[0], 8)
        self.assertEqual(mpr.U.LL.x.values[0], 6)

    def test_toyexample3(self):
        mpr = examples.toyexample3.create()
        mpr.check()

        opt = Solver('pao.mpr.MIBS')

        try:
            opt.solve(mpr)
            self.fail("Expected an error: linking variables should be integer")
        except RuntimeError as err:
            print("Solver run-time error:", err)

        # self.assertTrue(math.isclose(mpr.U.x.values[0], 3, abs_tol=1e-4))
        # self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 0.5, abs_tol=1e-4))
        # self.assertEqual(mpr.U.x.values[1], 8)
        # self.assertEqual(mpr.U.LL.x.values[1], 0)


#class Test_bilevel_ld(unittest.TestCase):
class XTest_bilevel_ld(object):

    def test_besancon27(self):
        mpr = examples.besancon27.create()
        mpr.check()
        mpr.print()

        opt = Solver('pao.bilevel.ld')
        opt.solve(mpr)

        self.assertTrue(math.isclose(mpr.U.x.values[0], 0))
        self.assertTrue(math.isclose(mpr.U.LL.x.values[0], 1))

if __name__ == "__main__":
    unittest.main()
