#import numpy as np
#import scipy.sparse
import pyutilib.th as unittest
from pao.tensor import *
from pao.tensor.convert_repn import convert_LinearBilevelProblem_to_standard_form


class Test_Trivial(unittest.TestCase):

    def test_trivial1(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(len(ans.U.b), len(blp.U.b))

    def test_trivial2(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.c.U.xR = [1]
        U.c.U.xZ = [1, 1]
        U.c.U.xB = [1, 1, 1]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(len(ans.U.b), len(blp.U.b))

    def test_trivial3(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        #   with lower-level variables
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.c.U.xR = [1]
        U.c.U.xZ = [1, 1]
        U.c.U.xB = [1, 1, 1]
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0]))
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(len(ans.L.b), len(blp.L.b))

    def test_trivial4(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        #   with lower-level variables
        #   with lower-level objective
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.c.U.xR = [1]
        U.c.U.xZ = [1, 1]
        U.c.U.xB = [1, 1, 1]
        U.c.L.xR = [1, 1]
        U.c.L.xZ = [1, 1, 1]
        U.c.L.xB = [1, 1, 1, 1]
        L = blp.add_lower(nxR=2, nxZ=3, nxB=4)
        L.c.U.xR = [1]
        L.c.U.xZ = [1, 1]
        L.c.U.xB = [1, 1, 1]
        L.c.L.xR = [1, 1]
        L.c.L.xZ = [1, 1, 1]
        L.c.L.xB = [1, 1, 1, 1]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(len(ans.U.b), len(blp.U.b))

        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U))
        self.assertEqual(len(ans.L.A.U), len(blp.L.A.U))
        self.assertEqual(len(ans.L.b), len(blp.L.b))


class Test_Upper(unittest.TestCase):

    def test_test3(self):
        # Expect Changes - Nontrivial problem
        #   upper-level inequality constraints, so slack variables should be added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.xR.lower_bounds = [0]
        U.A.U.xR = [(0,0,1)]
        U.b = [2]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

    def test_test3L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level inequality constraints, so slack variables should be added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.xR.lower_bounds = [0]
        U.A.U.xR = [(0,0,1)]
        U.b = [2]
        L = blp.add_lower(nxZ=2, nxB=3)
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

    def test_test4(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3]
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(ans.U.c.U.xR[0], 2)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

    def test_test4L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3]
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(ans.U.c.U.xR[0], 2)
        self.assertEqual(ans.L.d, 27)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0]))
        self.assertEqual(ans.L.c.U.xR[0], 9)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

    def test_test5(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.upper_bounds = [3]
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(ans.U.c.U.xR[0], -2)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], -5)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

    def test_test5L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.upper_bounds = [3]
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U))
        self.assertEqual(ans.U.c.U.xR[0], -2)
        self.assertEqual(ans.L.d, 27)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0]))
        self.assertEqual(ans.L.c.U.xR[0], -9)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], -5)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

    def test_test6(self):
        # Expect Changes - Nontrivial problem
        #   upper-level range bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3]
        U.xR.upper_bounds = [9]
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        blp.check()
        #blp.print()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(ans.U.c.U.xR[0], 2)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[1,0], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(ans.U.b[1], 6)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

    def test_test6L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level range bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3]
        U.xR.upper_bounds = [9]
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9]
        blp.check()
        #blp.print()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(ans.U.c.U.xR[0], 2)
        self.assertEqual(ans.U.c.U.xR[1], 0)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[1,0], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(ans.U.b[1], 6)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

        self.assertEqual(ans.L.d, 27)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+1)
        self.assertEqual(ans.L.c.U.xR[0], 9)
        self.assertEqual(ans.L.c.U.xR[1], 0)

    def test_test7(self):
        # Expect Changes - Nontrivial problem
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        blp.check()
        #blp.print()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(ans.U.c.U.xR[0], 2)
        self.assertEqual(ans.U.c.U.xR[1], -2)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], -5)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

    def test_test7L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = False
        U.c.U.xR = [2]
        U.A.U.xR = [(0,0,5)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9]
        blp.check()
        #blp.print()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(ans.U.c.U.xR[0], 2)
        self.assertEqual(ans.U.c.U.xR[1], -2)
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], -5)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+1)
        self.assertEqual(ans.L.c.U.xR[0], 9)
        self.assertEqual(ans.L.c.U.xR[1], -9)

    def test_test8(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      ]
        U.xR.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF]
        U.c.U.xR = [2, 3, 4, 5, 6]
        U.A.U.xR = [(0,0,5), (0,1,17), (0,2,19), (0,3,23), (0,4,29)]
        U.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+2)
        self.assertEqual(list(ans.U.c.U.xR), [2,-3,4,5,6,0,-5])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], -17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], 23)
        self.assertEqual(ans.U.A.U.xR.todok()[0,4], 29)
        self.assertEqual(ans.U.A.U.xR.todok()[0,6], -23)
        self.assertEqual(ans.U.A.U.xR.todok()[1,5], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+2)

    def test_test8L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      ]
        U.xR.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF]
        U.c.U.xR = [2, 3, 4, 5, 6]
        U.A.U.xR = [(0,0,5), (0,1,17), (0,2,19), (0,3,23), (0,4,29)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9,10,11,12,13]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+2)
        self.assertEqual(list(ans.U.c.U.xR), [2,-3,4,5,6,0,-5])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], -17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], 23)
        self.assertEqual(ans.U.A.U.xR.todok()[0,4], 29)
        self.assertEqual(ans.U.A.U.xR.todok()[0,6], -23)
        self.assertEqual(ans.U.A.U.xR.todok()[1,5], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+2)

        self.assertEqual(ans.L.d, 238)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+2)
        self.assertEqual(list(ans.L.c.U.xR), [9,-10,11,12,13,0,-12])

    def test_test9(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.inequalities = True
        U.xR.lower_bounds = [0, 0, 0]
        U.c.U.xR = [2, 3, 4]
        U.A.U.xR = [(0,0,5), (1,1,17), (2,2,19)]
        U.b = [7,8,9]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+3)
        self.assertEqual(list(ans.U.c.U.xR), [2,3,4,0,0,0])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[1,1], 17)
        self.assertEqual(ans.U.A.U.xR.todok()[2,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[1,4], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[2,5], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(ans.U.b[1], 8)
        self.assertEqual(ans.U.b[2], 9)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+3)

    def test_test9L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.inequalities = True
        U.xR.lower_bounds = [0, 0, 0]
        U.c.U.xR = [2, 3, 4]
        U.A.U.xR = [(0,0,5), (1,1,17), (2,2,19)]
        U.b = [7,8,9]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9,10,11]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+3)
        self.assertEqual(list(ans.U.c.U.xR), [2,3,4,0,0,0])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[1,1], 17)
        self.assertEqual(ans.U.A.U.xR.todok()[2,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[1,4], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[2,5], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(ans.U.b[1], 8)
        self.assertEqual(ans.U.b[2], 9)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+3)

        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+3)
        self.assertEqual(list(ans.L.c.U.xR), [9,10,11,0,0,0])

    def test_test10(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3, np.NINF, 0]
        U.c.U.xR = [2, 3, 4]
        U.A.U.xR = [(0,0,5), (0,1,17), (0,2,19)]
        U.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(list(ans.U.c.U.xR), [2,3,4,-3])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], 17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], -17)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

    def test_test10L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.lower_bounds = [3, np.NINF, 0]
        U.c.U.xR = [2, 3, 4]
        U.A.U.xR = [(0,0,5), (0,1,17), (0,2,19)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9,10,11]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(list(ans.U.c.U.xR), [2,3,4,-3])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], 17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], -17)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

        self.assertEqual(ans.L.d, 27)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+1)
        self.assertEqual(list(ans.L.c.U.xR), [9,10,11,-10])

    def test_test11(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=2, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.upper_bounds = [3, np.PINF]
        U.c.U.xR = [2, 3]
        U.A.U.xR = [(0,0,5), (0,1,17)]
        U.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(list(ans.U.c.U.xR), [-2,3,-3])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], -5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], 17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], -17)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

    def test_test11L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=2, nxZ=2, nxB=3)
        U.inequalities = False
        U.xR.upper_bounds = [3, np.PINF]
        U.c.U.xR = [2, 3]
        U.A.U.xR = [(0,0,5), (0,1,17)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9,10]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+1)
        self.assertEqual(list(ans.U.c.U.xR), [-2,3,-3])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U))
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], -5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], 17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], -17)
        self.assertEqual(len(ans.U.b), len(blp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+1)

        self.assertEqual(ans.L.d, 27)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+1)
        self.assertEqual(list(ans.L.c.U.xR), [-9,10,-10])

    def test_test12(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.inequalities = True
        U.xR.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      ]
        U.xR.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF]
        U.c.U.xR = [2, 3, 4, 5, 6]
        U.A.U.xR = [(0,0,5), (0,1,17), (0,2,19), (0,3,23), (0,4,29)]
        U.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+3)
        self.assertEqual(list(ans.U.c.U.xR), [2,-3,4,5,6,-5,0,0])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], -17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], 23)
        self.assertEqual(ans.U.A.U.xR.todok()[0,4], 29)
        self.assertEqual(ans.U.A.U.xR.todok()[0,5], -23)
        self.assertEqual(ans.U.A.U.xR.todok()[1,2], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,6], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[1,7], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+3)

    def test_test12L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.inequalities = True
        U.xR.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      ]
        U.xR.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF]
        U.c.U.xR = [2, 3, 4, 5, 6]
        U.A.U.xR = [(0,0,5), (0,1,17), (0,2,19), (0,3,23), (0,4,29)]
        U.b = [7]
        L = blp.add_lower(nxZ=2, nxB=3)
        L.c.U.xR = [9,10,11,12,13]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+3)
        self.assertEqual(list(ans.U.c.U.xR), [2,-3,4,5,6,-5,0,0])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        self.assertEqual(ans.U.A.U.xR.todok()[0,1], -17)
        self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        self.assertEqual(ans.U.A.U.xR.todok()[0,3], 23)
        self.assertEqual(ans.U.A.U.xR.todok()[0,4], 29)
        self.assertEqual(ans.U.A.U.xR.todok()[0,5], -23)
        self.assertEqual(ans.U.A.U.xR.todok()[1,2], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[0,6], 1)
        self.assertEqual(ans.U.A.U.xR.todok()[1,7], 1)
        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+3)

        self.assertEqual(ans.L.d, 238)
        self.assertEqual(len(ans.L.c.U), len(blp.L.c.U)+3)
        self.assertEqual(list(ans.L.c.U.xR), [9,-10,11,12,13,-12,0,0])


class Test_Lower(unittest.TestCase):

    def test_test3(self):
        # Expect Changes - Nontrivial problem
        #   lower-level inequality constraints, so slack variables should be added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        L.xR.lower_bounds = [0]
        L.A.L[0].xR = [(0,0,1)]
        L.b = [2]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0]))
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+1)

    def test_test4(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        L.inequalities = False
        L.xR.lower_bounds = [3]
        L.c.L.xR = [2]
        L.A.L.xR = [(0,0,5)]
        L.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 6)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0]))
        self.assertEqual(ans.L.c.L.xR[0], 2)
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(ans.L.b[0], -8)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR))

    def test_test5(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        L.inequalities = False
        L.xR.upper_bounds = [3]
        L.c.L.xR = [2]
        L.A.L.xR = [(0,0,5)]
        L.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 6)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0]))
        self.assertEqual(ans.L.c.L.xR[0], -2)
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], -5)
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(ans.L.b[0], -8)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR))

    def test_test6(self):
        # Expect Changes - Nontrivial problem
        #   upper-level range bounds, so the const objective and RHS are changed
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        L.inequalities = False
        L.xR.lower_bounds = [3]
        L.xR.upper_bounds = [9]
        L.c.L.xR = [2]
        L.A.L.xR = [(0,0,5)]
        L.b = [7]
        blp.check()
        #blp.print()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 6)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+1)
        self.assertEqual(ans.L.c.L.xR[0], 2)
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0])+1)
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(ans.L.A.L.xR.todok()[1,0], 1)
        self.assertEqual(len(ans.L.b), len(blp.L.b)+1)
        self.assertEqual(ans.L.b[0], -8)
        self.assertEqual(ans.L.b[1], 6)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+1)

    def test_test7(self):
        # Expect Changes - Nontrivial problem
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        L.inequalities = False
        L.c.L.xR = [2]
        L.A.L.xR = [(0,0,5)]
        L.b = [7]
        blp.check()
        #blp.print()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+1)
        self.assertEqual(ans.L.c.L.xR[0], 2)
        self.assertEqual(ans.L.c.L.xR[1], -2)
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(ans.L.A.L.xR.todok()[0,1], -5)
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(ans.L.b[0], 7)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+1)

    def test_test8(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=5, nxZ=2, nxB=3)
        L.inequalities = False
        L.xR.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      ]
        L.xR.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF]
        L.c.L.xR = [2, 3, 4, 5, 6]
        L.A.L.xR = [(0,0,5), (0,1,17), (0,2,19), (0,3,23), (0,4,29)]
        L.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 77)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+2)
        self.assertEqual(list(ans.L.c.L.xR), [2,-3,4,5,6,0,-5])
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0])+1)
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(ans.L.A.L.xR.todok()[0,1], -17)
        self.assertEqual(ans.L.A.L.xR.todok()[0,2], 19)
        self.assertEqual(ans.L.A.L.xR.todok()[0,3], 23)
        self.assertEqual(ans.L.A.L.xR.todok()[0,4], 29)
        self.assertEqual(ans.L.A.L.xR.todok()[0,6], -23)
        self.assertEqual(ans.L.A.L.xR.todok()[1,5], 1)
        self.assertEqual(len(ans.L.b), len(blp.L.b)+1)
        self.assertEqual(ans.L.b[0], -370)
        self.assertEqual(ans.L.b[1], 2)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+2)

    def test_test9(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=3, nxZ=2, nxB=3)
        L.inequalities = True
        L.xR.lower_bounds = [0, 0, 0]
        L.c.L.xR = [2, 3, 4]
        L.A.L.xR = [(0,0,5), (1,1,17), (2,2,19)]
        L.b = [7,8,9]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 0)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+3)
        self.assertEqual(list(ans.L.c.L.xR), [2,3,4,0,0,0])
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(ans.L.A.L.xR.todok()[1,1], 17)
        self.assertEqual(ans.L.A.L.xR.todok()[2,2], 19)
        self.assertEqual(ans.L.A.L.xR.todok()[0,3], 1)
        self.assertEqual(ans.L.A.L.xR.todok()[1,4], 1)
        self.assertEqual(ans.L.A.L.xR.todok()[2,5], 1)
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(ans.L.b[0], 7)
        self.assertEqual(ans.L.b[1], 8)
        self.assertEqual(ans.L.b[2], 9)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+3)

    def test_test10(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=3, nxZ=2, nxB=3)
        L.inequalities = False
        L.xR.lower_bounds = [3, np.NINF, 0]
        L.c.L.xR = [2, 3, 4]
        L.A.L.xR = [(0,0,5), (0,1,17), (0,2,19)]
        L.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 6)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+1)
        self.assertEqual(list(ans.L.c.L.xR), [2,3,4,-3])
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(ans.L.A.L.xR.todok()[0,1], 17)
        self.assertEqual(ans.L.A.L.xR.todok()[0,2], 19)
        self.assertEqual(ans.L.A.L.xR.todok()[0,3], -17)
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(ans.L.b[0], -8)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+1)

    def test_test11(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=2, nxZ=2, nxB=3)
        L.inequalities = False
        L.xR.upper_bounds = [3, np.PINF]
        L.c.L.xR = [2, 3]
        L.A.L.xR = [(0,0,5), (0,1,17)]
        L.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+1)
        self.assertEqual(list(ans.L.c.L.xR), [-2,3,-3])
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0]))
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], -5)
        self.assertEqual(ans.L.A.L.xR.todok()[0,1], 17)
        self.assertEqual(ans.L.A.L.xR.todok()[0,2], -17)
        self.assertEqual(len(ans.L.b), len(blp.L.b))
        self.assertEqual(ans.L.b[0], -8)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+1)

    def test_test12(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxZ=2, nxB=3)
        L = blp.add_lower(nxR=5, nxZ=2, nxB=3)
        L.inequalities = True
        L.xR.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      ]
        L.xR.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF]
        L.c.L.xR = [2, 3, 4, 5, 6]
        L.A.L.xR = [(0,0,5), (0,1,17), (0,2,19), (0,3,23), (0,4,29)]
        L.b = [7]
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR))

        self.assertEqual(ans.L.d, 77)
        self.assertEqual(len(ans.L.c.L[0]), len(blp.L.c.L[0])+3)
        self.assertEqual(list(ans.L.c.L.xR), [2,-3,4,5,6,-5,0,0])
        self.assertEqual(len(ans.L.A.L[0]), len(blp.L.A.L[0])+1)
        self.assertEqual(ans.L.A.L.xR.todok()[0,0], 5)
        self.assertEqual(ans.L.A.L.xR.todok()[0,1], -17)
        self.assertEqual(ans.L.A.L.xR.todok()[0,2], 19)
        self.assertEqual(ans.L.A.L.xR.todok()[0,3], 23)
        self.assertEqual(ans.L.A.L.xR.todok()[0,4], 29)
        self.assertEqual(ans.L.A.L.xR.todok()[0,5], -23)
        self.assertEqual(ans.L.A.L.xR.todok()[1,2], 1)
        self.assertEqual(ans.L.A.L.xR.todok()[0,6], 1)
        self.assertEqual(ans.L.A.L.xR.todok()[1,7], 1)
        self.assertEqual(len(ans.L.b), len(blp.L.b)+1)
        self.assertEqual(ans.L.b[0], -370)
        self.assertEqual(ans.L.b[1], 2)
        self.assertEqual(len(ans.L.xR), len(blp.L.xR)+3)


class Test_NonTrivial(unittest.TestCase):

    def test_test1(self):
        blp = LinearBilevelProblem()

        U = blp.add_upper(nxR=4)
        blp.add_lower(nxR=5)
        L = blp.add_lower(nxR=6)

        U.xR.lower_bounds    = [3,       np.NINF, 7, np.NINF]
        U.xR.upper_bounds    = [np.PINF, 5,       11, np.PINF]

        L[0].xR.lower_bounds = [5,       np.NINF, 11, np.NINF, 0]
        L[0].xR.upper_bounds = [np.PINF, 7,       13, np.PINF, np.PINF]

        L[1].xR.lower_bounds = [0,       7,       np.NINF, 13, np.NINF, 0]
        L[1].xR.upper_bounds = [np.PINF, np.PINF, 11,      17, np.PINF, np.PINF]


        U.c.U.xR    = [2, 3, 4, 5]
        U.c.L[0].xR = [3, 4, 5, 6, 7]
        U.c.L[1].xR = [4, 5, 6, 7, 8, 9]

        U.inequalities = True
        U.A.U.xR =    [(0,0,1), (0,1,1), (0,2,1), (0,3,1),
                       (1,0,2),          (1,2,2),
                                (2,1,3),          (2,3,3)
                      ]
        U.A.L[0].xR = [(0,0,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1),
                       (1,0,2),          (1,2,2),
                                (2,1,3),          (2,3,3)
                      ]
        U.A.L[1].xR = [(0,0,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1), (0,5,1),
                       (1,0,2),          (1,2,2),
                                (2,1,3),          (2,3,3)
                      ]
        U.b = [2,3,5]


        L[0].c.U.xR    = [5, 6, 7, 8]
        L[0].c.L[0].xR = [6, 7, 8, 9, 10]

        L[0].inequalities = True
        L[0].A.U.xR =    [(0,0,2), (0,1,2), (0,2,2), (0,3,2),
                          (1,0,3),          (1,2,3),
                                   (2,1,4),          (2,3,4),
                                            (3,2,5)
                         ]
        L[0].A.L[0].xR = [(0,0,2), (0,1,2), (0,2,2), (0,3,2), (0,4,2),
                          (1,0,3),          (1,2,3),
                                   (2,1,4),          (2,3,4),
                                            (3,2,5),          (3,4,5)
                         ]
        L[0].b = [2,3,5,7]

        L[1].c.U.xR    = [5, 6, 7, 8]
        L[1].c.L[1].xR = [7, 8, 9, 10, 11, 12]

        L[1].inequalities = False
        L[1].A.U.xR =    [(1,0,3),          (1,2,3),
                                   (2,1,4),          (2,3,4),
                                            (3,2,5)
                         ]
        L[1].A.L[1].xR = [(0,0,2), (0,1,2), (0,2,2), (0,3,2), (0,4,2), (0,5,2),
                          (1,0,3),          (1,2,3),
                                   (2,1,4),          (2,3,4),
                                            (3,2,5),          (3,4,5)
                         ]
        L[1].b = [1,2,3,5]

        #blp.print()
        blp.check()

        #print("-"*80)

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        # Just some sanity checks here
        self.assertEqual(ans.U.d, 339)
        self.assertEqual(ans.L[0].d, 261)
        self.assertEqual(ans.L[1].d, 379)

        self.assertEqual(list(ans.U.c.U.xR),        [ 2, -3,  4,  5, -5,  0, 0,  0, 0])
        self.assertEqual(list(ans.U.c.L[0].xR),     [ 3, -4,  5,  6,  7, -6, 0,  0, 0, 0, 0])
        self.assertEqual(list(ans.U.c.L[1].xR),     [ 4,  5, -6,  7,  8,  9, 0, -8])
        self.assertEqual(list(ans.L[0].c.U.xR),     [ 5, -6,  7,  8, -8,  0, 0,  0, 0])
        self.assertEqual(list(ans.L[0].c.L[0].xR),  [ 6, -7,  8,  9, 10, -9, 0,  0, 0, 0, 0])
        self.assertEqual(list(ans.L[1].c.U.xR),     [ 5, -6,  7,  8, -8,  0, 0,  0, 0])
        self.assertEqual(list(ans.L[1].c.L[1].xR),  [ 7,  8, -9, 10, 11, 12, 0, -11])

        self.assertEqual(list(ans.U.b),       [-67, -71, -91, 4.])
        self.assertEqual(list(ans.L[0].b),    [-74, -75, -43, -83, 2])
        self.assertEqual(list(ans.L[1].b),    [-61, -61, -97, -85,  4])


class Test_Examples(unittest.TestCase):

    def test_simple1(self):
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=6, nxZ=0, nxB=0)
        U.inequalities = False
        U.xR.lower_bounds = [np.NINF, -1,      np.NINF, -2, 0,       0]
        U.xR.upper_bounds = [np.PINF, np.PINF, 5,        2, np.PINF, np.PINF]
        U.c.U.xR = [3, -1, 0, 0, 0, 0]
        U.A.U.xR = [(0,0,-1), (0,1,6), (0,2,-1), (0,3,1), (0,4,-1),
                              (1,1,7),           (1,3,1),
                                       (2,2,1),  (2,3,1),           (2,5,1)]
        U.b = [-3, 5, 2]
        #blp.print()
        blp.check()

        ans = convert_LinearBilevelProblem_to_standard_form(blp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 1)
        self.assertEqual(len(ans.U.c.U), len(blp.U.c.U)+2)
        self.assertEqual(list(ans.U.c.U.xR), [3, -1, 0, 0, 0, 0, -3, 0])
        self.assertEqual(len(ans.U.A.U), len(blp.U.A.U)+1)

        #self.assertEqual(ans.U.A.U.xR.todok()[0,0], 5)
        #self.assertEqual(ans.U.A.U.xR.todok()[0,1], -17)
        #self.assertEqual(ans.U.A.U.xR.todok()[0,2], 19)
        #self.assertEqual(ans.U.A.U.xR.todok()[0,3], 23)
        #self.assertEqual(ans.U.A.U.xR.todok()[0,4], 29)
        #self.assertEqual(ans.U.A.U.xR.todok()[0,5], -23)
        #self.assertEqual(ans.U.A.U.xR.todok()[1,2], 1)
        #self.assertEqual(ans.U.A.U.xR.todok()[0,6], 1)
        #self.assertEqual(ans.U.A.U.xR.todok()[1,7], 1)

        self.assertEqual(len(ans.U.b), len(blp.U.b)+1)
        self.assertEqual(ans.U.b[0], 10)
        self.assertEqual(ans.U.b[1], 14)
        self.assertEqual(ans.U.b[2], -1)
        self.assertEqual(ans.U.b[3], 4)
        self.assertEqual(len(ans.U.xR), len(blp.U.xR)+2)


if __name__ == "__main__":
    unittest.main()
