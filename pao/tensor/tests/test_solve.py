import math
import pyutilib.th as unittest
from pao.tensor import *
from pao.tensor import examples


class Test_bilevel_blp(unittest.TestCase):

    def test_bard511(self):
        blp = examples.bard511.create()
        blp.check()

        opt = LinearBilevelSolver('pao.bilevel.blp_global')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_bard511_L0(self):
        blp = examples.bard511_L0.create()
        blp.check()

        opt = LinearBilevelSolver('pao.bilevel.blp_global')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def Xtest_bard511_L1(self):
        blp = examples.bard511_L1.create()
        blp.check()

        opt = LinearBilevelSolver('pao.bilevel.blp_global')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_bard511_list_L0(self):
        blp = examples.bard511_list_L0.create()
        blp.check()

        opt = LinearBilevelSolver('pao.bilevel.blp_global')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_bard511_list(self):
        blp = examples.bard511_list.create()
        blp.check()

        opt = LinearBilevelSolver('pao.bilevel.blp_global')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_besancon27(self):
        blp = examples.besancon27.create()
        blp.check()

        opt = LinearBilevelSolver('pao.bilevel.blp_global')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 0))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 1))


class Test_bilevel_FA(unittest.TestCase):

    def test_bard511(self):
        blp = examples.bard511.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_bard511_L0(self):
        blp = examples.bard511_L0.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def Xtest_bard511_L1(self):
        blp = examples.bard511_L1.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_bard511_list_L0(self):
        blp = examples.bard511_list_L0.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_bard511_list(self):
        blp = examples.bard511_list.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4))

    def test_besancon27(self):
        blp = examples.besancon27.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 0))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 1))

    def test_getachew_ex1(self):
        blp = examples.getachew_ex1.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 8))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 6))

    def test_getachew_ex2(self):
        blp = examples.getachew_ex2.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.FA')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 6))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 8))


class Test_bilevel_REG(unittest.TestCase):

    def test_bard511(self):
        blp = examples.bard511.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4, abs_tol=1e-4))

    def test_bard511_L0(self):
        blp = examples.bard511_L0.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4, abs_tol=1e-4))

    def Xtest_bard511_L1(self):
        blp = examples.bard511_L1.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4, abs_tol=1e-4))

    def test_bard511_list_L0(self):
        blp = examples.bard511_list_L0.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4, abs_tol=1e-4))

    def test_bard511_list(self):
        blp = examples.bard511_list.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 4, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 4, abs_tol=1e-4))

    def test_besancon27(self):
        blp = examples.besancon27.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 0, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 1, abs_tol=1e-4))

    def test_getachew_ex1(self):
        blp = examples.getachew_ex1.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 8, abs_tol=1e-4))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 6, abs_tol=1e-4))

    def test_getachew_ex2(self):
        blp = examples.getachew_ex2.create()
        blp.check()

        opt = LinearBilevelSolver('pao.lbp.REG')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 6))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 8))


#class Test_bilevel_ld(unittest.TestCase):
class XTest_bilevel_ld(object):

    def test_besancon27(self):
        blp = examples.besancon27.create()
        blp.check()
        blp.print()

        opt = LinearBilevelSolver('pao.bilevel.ld')
        opt.solve(blp)

        self.assertTrue(math.isclose(blp.U.xR.values[0], 0))
        self.assertTrue(math.isclose(blp.L.xR.values[0], 1))

if __name__ == "__main__":
    unittest.main()
