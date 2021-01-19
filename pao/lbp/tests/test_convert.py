import numpy as np
import pyutilib.th as unittest
from pao.lbp import *
from pao.lbp.convert_repn import convert_LinearMultilevelProblem_to_standard_form, convert_binaries_to_integers


class Test_Trivial_Linear(unittest.TestCase):

    def _create(self):
        return LinearMultilevelProblem()

    def test_trivial1(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(ans.U.c[U], None)
        self.assertEqual(lbp.U.c[U], None)
        self.assertEqual(ans.U.A[U], None)
        self.assertEqual(lbp.U.A[U], None)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))

    def test_trivial1L(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        L.x.lower_bounds = [0]*6
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(ans.U.c[U], None)
        self.assertEqual(lbp.U.c[U], None)
        self.assertEqual(ans.U.A[U], None)
        self.assertEqual(lbp.U.A[U], None)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))

    def test_trivial2(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.c[U] = [1, 1, 1, 1, 1, 1]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()
        #print("LMP")
        #lbp.print()
        #print("ANS")
        #ans.print()

        #print(id(lbp.U), id(ans.U))
        #print(id(lbp.U.x), id(ans.U.x))
        #print(id(lbp.U.c[U]), id(ans.U.c[U]))

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(ans.U.A[U], None)
        self.assertEqual(lbp.U.A[U], None)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))

    def test_trivial2_max(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        U.minimize = False
        U.c[U] = [1, 1, 1, 1, 1, 1]
        U.c[L] = [2, 2, 2, 2, 2, 2]
        L.minimize = False
        L.c[U] = [3, 3, 3, 3, 3, 3]
        L.c[L] = [4, 4, 4, 4, 4, 4]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(ans.U.A[U], None)
        self.assertEqual(lbp.U.A[U], None)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+3)

        self.assertEqual(list(ans.U.c[U]), [-1,1,-1,-1,1,1,-1,-1,-1])
        self.assertEqual(list(ans.U.c[L]), [-2,2,-2,-2,2,2,-2,-2,-2])

        self.assertEqual(list(ans.U.LL.c[U]), [-3,3,-3,-3,3,3,-3,-3,-3])
        self.assertEqual(list(ans.U.LL.c[L]), [-4,4,-4,-4,4,4,-4,-4,-4])

    def test_trivial3(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        #   with lower-level variables
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.c[U] = [1]*6
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(ans.U.A[U], None)
        self.assertEqual(lbp.U.A[U], None)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(ans.U.LL.c[L], None)
        self.assertEqual(lbp.U.LL.c[L], None)
        self.assertEqual(ans.U.LL.A[L], None)
        self.assertEqual(lbp.U.LL.A[L], None)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))

    def test_trivial4(self):
        # No changes are expected with a trivial problem
        #   with upper-level variables
        #   with upper-level objective
        #   with lower-level variables
        #   with lower-level objective
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=2, nxZ=3, nxB=4)
        U.c[U] = [1]*6
        U.c[L] = [1]*9
        L.c[U] = [1]*6
        L.c[L] = [1]*9
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(ans.U.A[U], None)
        self.assertEqual(lbp.U.A[U], None)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))

        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(ans.U.LL.A[U], None)
        self.assertEqual(lbp.U.LL.A[U], None)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))


class Test_Trivial_Quadratic(Test_Trivial_Linear):

    def _create(self):
        return QuadraticMultilevelProblem()


class Test_Upper_Linear(unittest.TestCase):

    def _create(self):
        return LinearMultilevelProblem()

    def test_test3(self):
        # Expect Changes - Nontrivial problem
        #   upper-level inequality constraints, so slack variables should be added
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.x.lower_bounds[0] = 0
        U.A[U] = [[1,0,0,0,0,0]]
        U.b = [2]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(ans.U.c[U], None)
        self.assertEqual(lbp.U.c[U], None)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

    def test_test3_inequality(self):
        # Expect Changes - Nontrivial problem
        #   upper-level equality constraints, so unconstrained variables are duplicated
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds[0] = 0
        U.A[U] = [[1,0,0,0,0,0]]
        U.b = [2]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp, inequalities=True)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(ans.U.c[U], None)
        self.assertEqual(lbp.U.c[U], None)
        self.assertEqual(ans.U.A[U].shape[0], 2*lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+2)
        self.assertEqual(len(ans.U.b), 2*len(lbp.U.b))
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

    def test_test3L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level inequality constraints, so slack variables should be added
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.x.lower_bounds[0] = 0
        U.A[U] = [[1,0,0,0,0,0]]
        U.b = [2]
        L = U.add_lower(nxZ=2, nxB=3)
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(ans.U.c[U], None)
        self.assertEqual(lbp.U.c[U], None)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

    def test_test4(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds[0] = 3
        U.x.lower_bounds[1] = 0
        U.x.lower_bounds[2] = 0
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U]))
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1])
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x))

    def test_test4L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds[0] = 3
        U.x.lower_bounds[1] = 0
        U.x.lower_bounds[2] = 0
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,0,0,0,0,0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U]))
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.LL.d, 27)
        self.assertEqual(ans.U.LL.c[L], None)
        self.assertEqual(lbp.U.LL.c[L], None)
        self.assertEqual(ans.U.LL.c[U][0], 9)
        self.assertEqual(ans.U.A[U].shape, lbp.U.A[U].shape)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x))

    def test_test4_inequality(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = True
        U.x.lower_bounds[0] = 3
        U.x.lower_bounds[1] = 0
        U.x.lower_bounds[2] = 0
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp, inequalities=True)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U]))
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.A[U].shape, lbp.U.A[U].shape)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x))

    def test_test5(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.upper_bounds = [3, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+2)
        self.assertEqual(ans.U.c[U][0], -2)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+2)
        self.assertEqual(ans.U.A[U].todok()[0,0], -5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

    def test_test5L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.upper_bounds[0] = 3
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,0,0,0,0,0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+2)
        self.assertEqual(ans.U.c[U][0], -2)
        self.assertEqual(ans.U.LL.d, 27)
        self.assertEqual(ans.U.LL.c[L], None)
        self.assertEqual(lbp.U.LL.c[L], None)
        self.assertEqual(ans.U.LL.c[U][0], -9)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+2)
        self.assertEqual(ans.U.A[U].todok()[0,0], -5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

    def test_test5_inequality(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.inequalities = True
        U.x.upper_bounds[0] = 3
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp, inequalities=True)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+2)
        self.assertEqual(ans.U.c[U][0], -2)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+2)
        self.assertEqual(ans.U.A[U].todok()[0,0], -5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

    def test_test6(self):
        # Expect Changes - Nontrivial problem
        #   upper-level range bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [3,0,0,0,0,0]
        U.x.upper_bounds = [9,np.PINF,np.PINF,1,1,1]
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        lbp.check()
        #lbp.print()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+1)
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+1)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[1,0], 1)
        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(ans.U.b[1], 6)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+1)

    def test_test6L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level range bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [3,0,0,0,0,0]
        U.x.upper_bounds = [9,np.PINF,np.PINF,1,1,1]
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,0,0,0,0,0]
        lbp.check()
        #lbp.print()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+1)
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.c[U][1], 0)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+1)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[1,0], 1)
        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(ans.U.b[1], 6)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+1)

        self.assertEqual(ans.U.LL.d, 27)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+1)
        self.assertEqual(ans.U.LL.c[U][0], 9)
        self.assertEqual(ans.U.LL.c[U][1], 0)

    def test_test7(self):
        # Expect Changes - Nontrivial problem
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        lbp.check()
        #lbp.print()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.c[U][1], -2)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], -5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

    def test_test7L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.equalities = True
        U.c[U] = [2,0,0,0,0,0]
        U.A[U] = [[5,0,0,0,0,0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,0,0,0,0,0]
        lbp.check()
        #lbp.print()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(ans.U.c[U][0], 2)
        self.assertEqual(ans.U.c[U][1], -2)
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], -5)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(ans.U.LL.c[U][0], 9)
        self.assertEqual(ans.U.LL.c[U][1], -9)

    def test_test8(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      , np.NINF, np.NINF, 0, 0, 0]
        U.x.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]
        U.A[U] = [[5, 17, 19, 23, 29, 0, 0, 0, 0, 0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+4)
        self.assertEqual(list(ans.U.c[U]), [2,-3,4,5,6,0,-5, 0,0,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+4)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], -17)
        self.assertEqual(ans.U.A[U].todok()[0,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], 23)
        self.assertEqual(ans.U.A[U].todok()[0,4], 29)
        self.assertEqual(ans.U.A[U].todok()[0,6], -23)
        self.assertEqual(ans.U.A[U].todok()[1,5], 1)
        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+4)

    def test_test8L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      , np.NINF, np.NINF, 0, 0, 0]
        U.x.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]
        U.A[U] = [[5, 17, 19, 23, 29, 0, 0, 0, 0, 0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,10,11,12,13,0,0,0,0,0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+4)
        self.assertEqual(list(ans.U.c[U]), [2,-3,4,5,6,0,-5,0,0,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+4)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], -17)
        self.assertEqual(ans.U.A[U].todok()[0,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], 23)
        self.assertEqual(ans.U.A[U].todok()[0,4], 29)
        self.assertEqual(ans.U.A[U].todok()[0,6], -23)
        self.assertEqual(ans.U.A[U].todok()[1,5], 1)
        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+4)

        self.assertEqual(ans.U.LL.d, 238)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+4)
        self.assertEqual(list(ans.U.LL.c[U]), [9,-10,11,12,13,0,-12,0,0,0,0,0,0,0])

    def test_test9(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.inequalities = True
        U.x.lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0]
        U.c[U] = [2, 3, 4, 0, 0, 0, 0, 0]
        U.A[U] = [[5,  0,  0, 0, 0, 0, 0, 0],
                  [0, 17,  0, 0, 0, 0, 0, 0],
                  [0,  0, 19, 0, 0, 0, 0, 0]]
        U.b = [7,8,9]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(list(ans.U.c[U]), [2,3,4,0,0,0,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[1,1], 17)
        self.assertEqual(ans.U.A[U].todok()[2,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], 1)
        self.assertEqual(ans.U.A[U].todok()[1,4], 1)
        self.assertEqual(ans.U.A[U].todok()[2,5], 1)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(ans.U.b[1], 8)
        self.assertEqual(ans.U.b[2], 9)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

    def test_test9L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.inequalities = True
        U.x.lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0]
        U.c[U] = [2, 3, 4, 0, 0, 0, 0, 0]
        U.A[U] = [[5,  0,  0, 0, 0, 0, 0, 0],
                  [0, 17,  0, 0, 0, 0, 0, 0],
                  [0,  0, 19, 0, 0, 0, 0, 0]]
        U.b = [7,8,9]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9, 10, 11, 0, 0, 0, 0, 0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(list(ans.U.c[U]), [2,3,4,0,0,0,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[1,1], 17)
        self.assertEqual(ans.U.A[U].todok()[2,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], 1)
        self.assertEqual(ans.U.A[U].todok()[1,4], 1)
        self.assertEqual(ans.U.A[U].todok()[2,5], 1)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(ans.U.b[1], 8)
        self.assertEqual(ans.U.b[2], 9)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(list(ans.U.LL.c[U]), [9,10,11,0,0,0,0,0,0,0,0])

    def test_test9_inequality(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0]
        U.c[U] = [2, 3, 4, 0, 0, 0, 0, 0]
        U.A[U] = [[5,  0,  0, 0, 0, 0, 0, 0],
                  [0, 17,  0, 0, 0, 0, 0, 0],
                  [0,  0, 19, 0, 0, 0, 0, 0]]
        U.b = [7,8,9]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp, inequalities=True)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U]))
        self.assertEqual(list(ans.U.c[U]), [2,3,4,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], 2*lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1])
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[1,1], 17)
        self.assertEqual(ans.U.A[U].todok()[2,2], 19)
        self.assertEqual(ans.U.A[U].todok()[3,0], -5)
        self.assertEqual(ans.U.A[U].todok()[4,1], -17)
        self.assertEqual(ans.U.A[U].todok()[5,2], -19)
        self.assertEqual(len(ans.U.b), 2*len(lbp.U.b))
        self.assertEqual(ans.U.b[0], 7)
        self.assertEqual(ans.U.b[1], 8)
        self.assertEqual(ans.U.b[2], 9)
        self.assertEqual(ans.U.b[3], -7)
        self.assertEqual(ans.U.b[4], -8)
        self.assertEqual(ans.U.b[5], -9)
        self.assertEqual(len(ans.U.x), len(lbp.U.x))

    def test_test10(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [3, np.NINF, 0, 0, 0, 0, 0, 0]
        U.c[U] = [2, 3, 4, 0, 0, 0, 0, 0]
        U.A[U] = [[5,17,19, 0, 0, 0, 0, 0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+1)
        self.assertEqual(list(ans.U.c[U]), [2,3,4,-3,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+1)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], 17)
        self.assertEqual(ans.U.A[U].todok()[0,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], -17)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+1)

    def test_test10L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=3, nxZ=2, nxB=3)
        U.equalities = True
        U.x.lower_bounds = [3, np.NINF, 0, 0, 0, 0, 0, 0]
        U.c[U] = [2, 3, 4, 0, 0, 0, 0, 0]
        U.A[U] = [[5,17,19, 0, 0, 0, 0, 0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,10,11, 0, 0, 0, 0, 0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+1)
        self.assertEqual(list(ans.U.c[U]), [2,3,4,-3,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+1)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], 17)
        self.assertEqual(ans.U.A[U].todok()[0,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], -17)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+1)

        self.assertEqual(ans.U.LL.d, 27)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+1)
        self.assertEqual(list(ans.U.LL.c[U]), [9,10,11,-10,0,0,0,0,0])

    def test_test11(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=2, nxZ=2, nxB=3)
        U.equalities = True
        U.x.upper_bounds = [3, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2, 3, 0, 0, 0, 0, 0]
        U.A[U] = [[5,17, 0, 0, 0, 0, 0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(list(ans.U.c[U]), [-2,3,-3,0,0,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], -5)
        self.assertEqual(ans.U.A[U].todok()[0,1], 17)
        self.assertEqual(ans.U.A[U].todok()[0,2], -17)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

    def test_test11L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=2, nxZ=2, nxB=3)
        U.equalities = True
        U.x.upper_bounds = [3, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2, 3, 0, 0, 0, 0, 0]
        U.A[U] = [[5,17, 0, 0, 0, 0, 0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,10, 0, 0, 0, 0, 0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 6)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(list(ans.U.c[U]), [-2,3,-3,0,0,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0])
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], -5)
        self.assertEqual(ans.U.A[U].todok()[0,1], 17)
        self.assertEqual(ans.U.A[U].todok()[0,2], -17)
        self.assertEqual(len(ans.U.b), len(lbp.U.b))
        self.assertEqual(ans.U.b[0], -8)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

        self.assertEqual(ans.U.LL.d, 27)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(list(ans.U.LL.c[U]), [-9,10,-10,0,0,0,0,0,0,0])

    def test_test12(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.inequalities = True
        U.x.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      , 0,       0,       0, 0, 0]
        U.x.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]
        U.A[U] = [[5,17,19,23,29, 0, 0, 0, 0, 0]]
        U.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(list(ans.U.c[U]), [2,-3,4,5,6,0,0,-5,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], -17)
        self.assertEqual(ans.U.A[U].todok()[0,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], 23)
        self.assertEqual(ans.U.A[U].todok()[0,4], 29)
        self.assertEqual(ans.U.A[U].todok()[0,5], 1)
        self.assertEqual(ans.U.A[U].todok()[1,2], 1)
        self.assertEqual(ans.U.A[U].todok()[1,6], 1)
        self.assertEqual(ans.U.A[U].todok()[0,7], -23)
        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

    def test_test12L(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxR=5, nxZ=2, nxB=3)
        U.inequalities = True
        U.x.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      , 0,       0,       0, 0, 0]
        U.x.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        U.c[U] = [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]
        U.A[U] = [[5,17,19,23,29, 0, 0, 0, 0, 0]]
        U.b = [7]
        L = U.add_lower(nxZ=2, nxB=3)
        L.c[U] = [9,10,11,12,13, 0, 0, 0, 0, 0]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 77)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(list(ans.U.c[U]), [2,-3,4,5,6,0,0,-5,0,0,0,0,0])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+3)
        self.assertEqual(ans.U.A[U].todok()[0,0], 5)
        self.assertEqual(ans.U.A[U].todok()[0,1], -17)
        self.assertEqual(ans.U.A[U].todok()[0,2], 19)
        self.assertEqual(ans.U.A[U].todok()[0,3], 23)
        self.assertEqual(ans.U.A[U].todok()[0,4], 29)
        self.assertEqual(ans.U.A[U].todok()[0,5], 1)
        self.assertEqual(ans.U.A[U].todok()[1,2], 1)
        self.assertEqual(ans.U.A[U].todok()[1,6], 1)
        self.assertEqual(ans.U.A[U].todok()[0,7], -23)
        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], -370)
        self.assertEqual(ans.U.b[1], 2)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)

        self.assertEqual(ans.U.LL.d, 238)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(list(ans.U.LL.c[U]), [9,-10,11,12,13,0,0,-12,0,0,0,0,0])


class Test_Upper_Quadratic(Test_Upper_Linear):

    def _create(self):
        return QuadraticMultilevelProblem()


class Test_Lower_Linear(unittest.TestCase):

    def _create(self):
        return LinearMultilevelProblem()

    def test_test3(self):
        # Expect Changes - Nontrivial problem
        #   lower-level inequality constraints, so slack variables should be added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        L.x.lower_bounds = [0, 0, 0, 0, 0, 0]
        L.A[L] = [[1, 0, 0, 0, 0, 0]]
        L.b = [2]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(ans.U.LL.c[L], None)
        self.assertEqual(lbp.U.LL.c[L], None)
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0])
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+1)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+1)

    def test_test4(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        L.equalities = True
        L.x.lower_bounds = [3,0,0,0,0,0]
        L.c[L] = [2,0,0,0,0,0]
        L.A[L] = [[5,0,0,0,0,0]]
        L.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L]))
        self.assertEqual(ans.U.LL.c[L][0], 2)
        self.assertEqual(ans.U.LL.A[L].shape, lbp.U.LL.A[L].shape)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(ans.U.LL.b[0], -8)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x))

    def test_test5(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        L.equalities = True
        L.x.upper_bounds = [3,0,0,0,0,0]
        L.c[L] = [2,0,0,0,0,0]
        L.A[L] = [[5,0,0,0,0,0]]
        L.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L]))
        self.assertEqual(ans.U.LL.c[L][0], -2)
        self.assertEqual(ans.U.LL.A[L].shape, lbp.U.LL.A[L].shape)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], -5)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(ans.U.LL.b[0], -8)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x))

    def test_test6(self):
        # Expect Changes - Nontrivial problem
        #   upper-level range bounds, so the const objective and RHS are changed
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        L.equalities = True
        L.x.lower_bounds = [3,0,0,0,0,0]
        L.x.upper_bounds = [9,np.PINF,np.PINF,1,1,1]
        L.c[L] = [2,0,0,0,0,0]
        L.A[L] = [[5,0,0,0,0,0]]
        L.b = [7]
        lbp.check()
        #lbp.print()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+1)
        self.assertEqual(ans.U.LL.c[L][0], 2)
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0]+1)
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+1)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(ans.U.LL.A[L].todok()[1,0], 1)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b)+1)
        self.assertEqual(ans.U.LL.b[0], -8)
        self.assertEqual(ans.U.LL.b[1], 6)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+1)

    def test_test7(self):
        # Expect Changes - Nontrivial problem
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)
        L.equalities = True
        L.c[L] = [2,0,0,0,0,0]
        L.A[L] = [[5,0,0,0,0,0]]
        L.b = [7]
        lbp.check()
        #lbp.print()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+3)
        self.assertEqual(ans.U.LL.c[L][0], 2)
        self.assertEqual(ans.U.LL.c[L][1], -2)
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0])
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+3)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(ans.U.LL.A[L].todok()[0,1], -5)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(ans.U.LL.b[0], 7)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+3)

    def test_test8(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=5, nxZ=2, nxB=3)
        L.equalities = True
        L.x.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      , 0,       0,       0, 0, 0]
        L.x.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        L.c[L] = [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]
        L.A[L] = [[5,17,19,23,29, 0, 0, 0, 0, 0]]
        L.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 77)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+2)
        self.assertEqual(list(ans.U.LL.c[L]), [2,-3,4,5,6,0,-5,0,0,0,0,0])
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0]+1)
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+2)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(ans.U.LL.A[L].todok()[0,1], -17)
        self.assertEqual(ans.U.LL.A[L].todok()[0,2], 19)
        self.assertEqual(ans.U.LL.A[L].todok()[0,3], 23)
        self.assertEqual(ans.U.LL.A[L].todok()[0,4], 29)
        self.assertEqual(ans.U.LL.A[L].todok()[0,6], -23)
        self.assertEqual(ans.U.LL.A[L].todok()[1,5], 1)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b)+1)
        self.assertEqual(ans.U.LL.b[0], -370)
        self.assertEqual(ans.U.LL.b[1], 2)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+2)

    def test_test9(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=3, nxZ=2, nxB=3)
        L.inequalities = True
        L.x.lower_bounds = [0, 0, 0, np.NINF, np.NINF, 0, 0, 0]
        L.c[L] = [2, 3, 4, 0, 0, 0, 0, 0]
        L.A[L] = [[5, 0,0 ,0,0,0,0,0],
                  [0,17,0 ,0,0,0,0,0],
                  [0, 0,19,0,0,0,0,0]]
        L.b = [7,8,9]
        #lbp.print()
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 0)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+5)
        self.assertEqual(list(ans.U.LL.c[L]), [2,3,4,0,0,0,0,0,0,0,0,0,0])
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0])
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+5)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(ans.U.LL.A[L].todok()[1,1], 17)
        self.assertEqual(ans.U.LL.A[L].todok()[2,2], 19)
        self.assertEqual(ans.U.LL.A[L].todok()[0,3], 1)
        self.assertEqual(ans.U.LL.A[L].todok()[1,4], 1)
        self.assertEqual(ans.U.LL.A[L].todok()[2,5], 1)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(ans.U.LL.b[0], 7)
        self.assertEqual(ans.U.LL.b[1], 8)
        self.assertEqual(ans.U.LL.b[2], 9)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+5)

    def test_test10(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=3, nxZ=2, nxB=3)
        L.equalities = True
        L.x.lower_bounds = [3, np.NINF, 0, 0, 0, 0, 0, 0]
        L.c[L] = [2, 3, 4, 0, 0, 0, 0, 0]
        L.A[L] = [[5,17,19, 0, 0, 0, 0, 0]]
        L.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+1)
        self.assertEqual(list(ans.U.LL.c[L]), [2,3,4,-3,0,0,0,0,0])
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0])
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+1)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(ans.U.LL.A[L].todok()[0,1], 17)
        self.assertEqual(ans.U.LL.A[L].todok()[0,2], 19)
        self.assertEqual(ans.U.LL.A[L].todok()[0,3], -17)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(ans.U.LL.b[0], -8)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+1)

    def test_test11(self):
        # Expect Changes - Nontrivial problem
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=2, nxZ=2, nxB=3)
        L.equalities = True
        L.x.upper_bounds = [3, np.PINF, 0, 0, 0, 0, 0]
        L.c[L] = [2, 3, 0, 0, 0, 0, 0]
        L.A[L] = [[5,17, 0, 0, 0, 0, 0]]
        L.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+1)
        self.assertEqual(list(ans.U.LL.c[L]), [-2,3,-3,0,0,0,0,0])
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0])
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+1)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], -5)
        self.assertEqual(ans.U.LL.A[L].todok()[0,1], 17)
        self.assertEqual(ans.U.LL.A[L].todok()[0,2], -17)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(ans.U.LL.b[0], -8)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+1)

    def test_test12(self):
        # Expect Changes - Nontrivial problem
        #   upper-level lower bounds, so the const objective and RHS are changed
        #   upper-level upper bounds, so the const objective and RHS are changed
        #   upper-level range bounds, so the const objective and RHS are changed
        #   upper-level unbounded variables, so variables are added
        lbp = self._create()
        U = lbp.add_upper(nxZ=2, nxB=3)
        L = U.add_lower(nxR=5, nxZ=2, nxB=3)
        L.inequalities = True
        L.x.lower_bounds = [3,       np.NINF, 11, np.NINF, 0      , 0      , 0      , 0, 0, 0]
        L.x.upper_bounds = [np.PINF, 9,       13, np.PINF, np.PINF, np.PINF, np.PINF, 1, 1, 1]
        L.c[L] = [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]
        L.A[L] = [[5,17,19,23,29, 0, 0, 0, 0, 0]]
        L.b = [7]
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        ans.check()

        self.assertEqual(ans.U.d, 0)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)

        self.assertEqual(ans.U.LL.d, 77)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+3)
        self.assertEqual(list(ans.U.LL.c[L]), [2,-3,4,5,6,0,0,-5, 0, 0, 0, 0 ,0])
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0]+1)
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+3)
        self.assertEqual(ans.U.LL.A[L].todok()[0,0], 5)
        self.assertEqual(ans.U.LL.A[L].todok()[0,1], -17)
        self.assertEqual(ans.U.LL.A[L].todok()[0,2], 19)
        self.assertEqual(ans.U.LL.A[L].todok()[0,3], 23)
        self.assertEqual(ans.U.LL.A[L].todok()[0,4], 29)
        self.assertEqual(ans.U.LL.A[L].todok()[0,5], 1)
        self.assertEqual(ans.U.LL.A[L].todok()[1,2], 1)
        self.assertEqual(ans.U.LL.A[L].todok()[1,6], 1)
        self.assertEqual(ans.U.LL.A[L].todok()[0,7], -23)
        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b)+1)
        self.assertEqual(ans.U.LL.b[0], -370)
        self.assertEqual(ans.U.LL.b[1], 2)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+3)


class Test_Lower_Quadratic(Test_Lower_Linear):

    def _create(self):
        return QuadraticMultilevelProblem()


class Test_NonTrivial_Linear(unittest.TestCase):

    def _create(self):
        return LinearMultilevelProblem()

    def test_test1(self):
        lbp = self._create()

        U = lbp.add_upper(nxR=4)
        L0 = U.add_lower(nxR=5)
        L1 = U.add_lower(nxR=6)

        U.x.lower_bounds  = [3,       np.NINF, 7, np.NINF]
        U.x.upper_bounds  = [np.PINF, 5,       11, np.PINF]

        L0.x.lower_bounds = [5,       np.NINF, 11, np.NINF, 0]
        L0.x.upper_bounds = [np.PINF, 7,       13, np.PINF, np.PINF]

        L1.x.lower_bounds = [0,       7,       np.NINF, 13, np.NINF, 0]
        L1.x.upper_bounds = [np.PINF, np.PINF, 11,      17, np.PINF, np.PINF]


        U.c[U]  = [2, 3, 4, 5]
        U.c[L0] = [3, 4, 5, 6, 7]
        U.c[L1] = [4, 5, 6, 7, 8, 9]

        U.inequalities = True
        U.A[U] =  [[1, 1, 1, 1],
                   [2, 0, 2, 0],
                   [0, 3, 0, 3]
                  ]
        U.A[L0] = [[1, 1, 1, 1, 1],
                   [2, 0, 2, 0, 0],
                   [0, 3, 0, 3, 0]
                  ]
        U.A[L1] = [[1, 1, 1, 1, 1, 1],
                   [2, 0, 2, 0, 0, 0],
                   [0, 3, 0, 3, 0, 0]
                  ]
        U.b = [2,3,5]


        L0.c[U]  = [5, 6, 7, 8]
        L0.c[L0] = [6, 7, 8, 9, 10]

        L0.inequalities = True
        L0.A[U] =     [[2, 2, 2, 2],
                       [3, 0, 3, 0],
                       [0, 4, 0, 4],
                       [0, 0, 5, 0]
                      ]
        L0.A[L0] = [[2, 2, 2, 2, 2],
                       [3, 0, 3, 0, 0],
                       [0, 4, 0, 4, 0],
                       [0, 0, 5, 0, 5]
                      ]
        L0.b = [2,3,5,7]

        L1.c[U]     = [5, 6, 7, 8]
        L1.c[L1] = [7, 8, 9, 10, 11, 12]

        L1.equalities = True
        L1.A[U] =     [[0, 0, 0, 0],
                       [3, 0, 3, 0],
                       [0, 4, 0, 4],
                       [0, 0, 5, 0]
                      ]
        L1.A[L1] = [[2, 2, 2, 2, 2, 2],
                       [3, 0, 3, 0, 0, 0],
                       [0, 4, 0, 4, 0, 0],
                       [0, 0, 5, 0, 5, 0]
                      ]
        L1.b = [1,2,3,5]

        #lbp.print()
        lbp.check()

        #print("-"*80)

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        # Just some sanity checks here
        self.assertEqual(ans.U.d, 339)
        self.assertEqual(ans.U.LL[0].d, 261)
        self.assertEqual(ans.U.LL[1].d, 379)

        self.assertEqual(list(ans.U.c[U]),         [ 2, -3,  4,  5,  0,  0, 0,  0, -5])
        self.assertEqual(list(ans.U.c[L0]),        [ 3, -4,  5,  6,  7,  0, 0,  0, 0, 0, -6])
        self.assertEqual(list(ans.U.c[L1]),        [ 4,  5, -6,  7,  8,  9, 0, -8])
        self.assertEqual(list(ans.U.LL[0].c[U]),   [ 5, -6,  7,  8,  0,  0, 0,  0, -8])
        self.assertEqual(list(ans.U.LL[0].c[L0]),  [ 6, -7,  8,  9, 10,  0, 0,  0, 0, 0, -9])
        self.assertEqual(list(ans.U.LL[1].c[U]),   [ 5, -6,  7,  8,  0,  0, 0,  0, -8])
        self.assertEqual(list(ans.U.LL[1].c[L1]),  [ 7,  8, -9, 10, 11, 12, 0, -11])

        self.assertEqual(list(ans.U.b),          [-67, -71, -91, 4.])
        self.assertEqual(list(ans.U.LL[0].b),    [-74, -75, -43, -83, 2])
        self.assertEqual(list(ans.U.LL[1].b),    [-61, -61, -97, -85,  4])

        self.assertEqual(soln_manager.multipliers[U.id],  [[(0,1)], [(1,-1)], [(2,1)], [(3,1),(8,-1)]])
        self.assertEqual(soln_manager.multipliers[L0.id], [[(0,1)], [(1,-1)], [(2,1)], [(3,1),(10,-1)], [(4,1)]])
        self.assertEqual(soln_manager.multipliers[L1.id], [[(0,1)], [(1,1)], [(2,-1)], [(3,1)], [(4,1),(7,-1)], [(5,1)]])

    def test_test1_inequality(self):
        lbp = self._create()

        U = lbp.add_upper(nxR=4)
        L0 = U.add_lower(nxR=5)
        L1 = U.add_lower(nxR=6)

        U.x.lower_bounds  = [3,       np.NINF, 7, np.NINF]
        U.x.upper_bounds  = [np.PINF, 5,       11, np.PINF]

        L0.x.lower_bounds = [5,       np.NINF, 11, np.NINF, 0]
        L0.x.upper_bounds = [np.PINF, 7,       13, np.PINF, np.PINF]

        L1.x.lower_bounds = [0,       7,       np.NINF, 13, np.NINF, 0]
        L1.x.upper_bounds = [np.PINF, np.PINF, 11,      17, np.PINF, np.PINF]


        U.c[U]  = [2, 3, 4, 5]
        U.c[L0] = [3, 4, 5, 6, 7]
        U.c[L1] = [4, 5, 6, 7, 8, 9]

        U.equalities = True
        U.A[U] =   [[1, 1, 1, 1],
                    [2, 0, 2, 0],
                    [0, 3, 0, 3]
                   ]
        U.A[L0] = [[1, 1, 1, 1, 1],
                    [2, 0, 2, 0, 0],
                    [0, 3, 0, 3, 0]
                   ]
        U.A[L1] = [[1, 1, 1, 1, 1, 1],
                    [2, 0, 2, 0, 0, 0],
                    [0, 3, 0, 3, 0, 0]
                   ]
        U.b = [2,3,5]


        L0.c[U]  = [5, 6, 7, 8]
        L0.c[L0] = [6, 7, 8, 9, 10]

        L0.equalities = True
        L0.A[U] =   [[2, 2, 2, 2],
                       [3, 0, 3, 0],
                       [0, 4, 0, 4],
                       [0, 0, 5, 0]
                      ]
        L0.A[L0] = [[2, 2, 2, 2, 2],
                       [3, 0, 3, 0, 0],
                       [0, 4, 0, 4, 0],
                       [0, 0, 5, 0, 5]
                      ]
        L0.b = [2,3,5,7]

        L1.c[U]   = [5, 6, 7, 8]
        L1.c[L1] = [7, 8, 9, 10, 11, 12]

        L1.inequalities = True
        L1.A[U] =   [[0, 0, 0, 0],
                       [3, 0, 3, 0],
                       [0, 4, 0, 4],
                       [0, 0, 5, 0]
                      ]
        L1.A[L1] = [[2, 2, 2, 2, 2, 2],
                       [3, 0, 3, 0, 0, 0],
                       [0, 4, 0, 4, 0, 0],
                       [0, 0, 5, 0, 5, 0]
                      ]
        L1.b = [1,2,3,5]

        #lbp.print()
        lbp.check()

        #print("-"*80)

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp, inequalities=True)
        #ans.print()
        ans.check()

        # Just some sanity checks here
        self.assertEqual(ans.U.d, 339)
        self.assertEqual(ans.U.LL[0].d, 261)
        self.assertEqual(ans.U.LL[1].d, 379)

        self.assertEqual(list(ans.U.c[U]),         [ 2, -3,  4,  5,  -5])
        self.assertEqual(list(ans.U.c[L0]),        [ 3, -4,  5,  6,  7,  -6])
        self.assertEqual(list(ans.U.c[L1]),        [ 4,  5, -6,  7,  8,  9, -8])
        self.assertEqual(list(ans.U.LL[0].c[U]),   [ 5, -6,  7,  8,  -8])
        self.assertEqual(list(ans.U.LL[0].c[L0]),  [ 6, -7,  8,  9, 10,  -9])
        self.assertEqual(list(ans.U.LL[1].c[U]),   [ 5, -6,  7,  8,  -8])
        self.assertEqual(list(ans.U.LL[1].c[L1]),  [ 7,  8, -9, 10, 11, 12, -11])

        self.assertEqual(list(ans.U.b),       [-67, -71, -91, 67, 71, 91, 4.])
        self.assertEqual(list(ans.U.LL[0].b), [-74, -75, -43, -83, 74, 75, 43, 83, 2])
        self.assertEqual(list(ans.U.LL[1].b), [-61, -61, -97, -85, 4])

        self.assertEqual(soln_manager.multipliers[U.id],  [[(0,1)], [(1,-1)], [(2,1)], [(3,1),(4,-1)]])
        self.assertEqual(soln_manager.multipliers[L0.id], [[(0,1)], [(1,-1)], [(2,1)], [(3,1),(5,-1)], [(4,1)]])
        self.assertEqual(soln_manager.multipliers[L1.id], [[(0,1)], [(1,1)], [(2,-1)], [(3,1)], [(4,1),(6,-1)], [(5,1)]])

    def test_test2(self):
        lbp = self._create()

        U = lbp.add_upper(nxR=4)
        U.minimize = False
        L0 = U.add_lower(nxR=5)
        L0.minimize = False
        L1 = U.add_lower(nxR=6)

        U.x.lower_bounds  = [3,       np.NINF, 7, np.NINF]
        U.x.upper_bounds  = [np.PINF, 5,       11, np.PINF]

        L0.x.lower_bounds = [5,       np.NINF, 11, np.NINF, 0]
        L0.x.upper_bounds = [np.PINF, 7,       13, np.PINF, np.PINF]

        L1.x.lower_bounds = [0,       7,       np.NINF, 13, np.NINF, 0]
        L1.x.upper_bounds = [np.PINF, np.PINF, 11,      17, np.PINF, np.PINF]


        U.c[U]    = [2, 3, 4, 5]
        U.c[L0] = [3, 4, 5, 6, 7]
        U.c[L1] = [4, 5, 6, 7, 8, 9]

        U.inequalities = True
        U.A[U] =    [[1, 1, 1, 1],
                       [2, 0, 2, 0],
                       [0, 3, 0, 3]
                      ]
        U.A[L0] = [[1, 1, 1, 1, 1],
                       [2, 0, 2, 0, 0],
                       [0, 3, 0, 3, 0]
                      ]
        U.A[L1] = [[1, 1, 1, 1, 1, 1],
                       [2, 0, 2, 0, 0, 0],
                       [0, 3, 0, 3, 0, 0]
                      ]
        U.b = [2,3,5]


        L0.c[U]    = [5, 6, 7, 8]
        L0.c[L0] = [6, 7, 8, 9, 10]

        L0.inequalities = True
        L0.A[U] =    [[2, 2, 2, 2],
                          [3, 0, 3, 0],
                          [0, 4, 0, 4],
                          [0, 0, 5, 0]
                         ]
        L0.A[L0] = [[2, 2, 2, 2, 2],
                          [3, 0, 3, 0, 0],
                          [0, 4, 0, 4, 0],
                          [0, 0, 5, 0, 5]
                         ]
        L0.b = [2,3,5,7]

        L1.c[U]    = [5, 6, 7, 8]
        L1.c[L1] = [7, 8, 9, 10, 11, 12]

        L1.equalities = True
        L1.A[U] =    [[0, 0, 0, 0],
                          [3, 0, 3, 0],
                          [0, 4, 0, 4],
                          [0, 0, 5, 0]
                         ]
        L1.A[L1] = [[2, 2, 2, 2, 2, 2],
                          [3, 0, 3, 0, 0, 0],
                          [0, 4, 0, 4, 0, 0],
                          [0, 0, 5, 0, 5, 0]
                         ]
        L1.b = [1,2,3,5]

        #lbp.print()
        lbp.check()

        #print("-"*80)

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        # Just some sanity checks here
        self.assertEqual(ans.U.d, -339)
        self.assertEqual(ans.U.LL[0].d, -261)
        self.assertEqual(ans.U.LL[1].d, 379)

        self.assertEqual(list(ans.U.c[U]),         [-2,  3, -4, -5,  0, 0,  0, 0, 5])
        self.assertEqual(list(ans.U.c[L0]),        [-3,  4, -5, -6, -7,  0,  0, 0, 0, 0, 6])
        self.assertEqual(list(ans.U.c[L1]),        [-4, -5,  6, -7, -8, -9, 0,  8])
        self.assertEqual(list(ans.U.LL[0].c[U]),   [-5,  6, -7, -8,  0,  0, 0,  0, 8])
        self.assertEqual(list(ans.U.LL[0].c[L0]),  [-6,  7, -8, -9,-10,  0, 0,  0, 0, 0, 9])
        self.assertEqual(list(ans.U.LL[1].c[U]),   [ 5, -6,  7,  8,  0,  0, 0,  0, -8])
        self.assertEqual(list(ans.U.LL[1].c[L1]),  [ 7,  8, -9, 10, 11, 12, 0, -11])

        self.assertEqual(list(ans.U.b),          [-67, -71, -91, 4.])
        self.assertEqual(list(ans.U.LL[0].b),    [-74, -75, -43, -83, 2])
        self.assertEqual(list(ans.U.LL[1].b),    [-61, -61, -97, -85,  4])

        self.assertEqual(soln_manager.multipliers[U.id],  [[(0,1)], [(1,-1)], [(2,1)], [(3,1),(8,-1)]])
        self.assertEqual(soln_manager.multipliers[L0.id], [[(0,1)], [(1,-1)], [(2,1)], [(3,1),(10,-1)], [(4,1)]])
        self.assertEqual(soln_manager.multipliers[L1.id], [[(0,1)], [(1,1)], [(2,-1)], [(3,1)], [(4,1),(7,-1)], [(5,1)]])


class Test_NonTrivial_Quadratic(Test_NonTrivial_Linear):

    def _create(self):
        return QuadraticMultilevelProblem()


class Test_Integers_Linear(unittest.TestCase):

    def _create(self):
        return LinearMultilevelProblem()

    def test_test1(self):
        lbp = self._create()

        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=2, nxZ=3, nxB=4)

        U.x.lower_bounds    = [0, np.NINF, np.NINF, 0, 0, 0]
        U.x.upper_bounds    = [1, 5      , np.PINF, 1, 1, 1]

        L.x.lower_bounds = [5,       np.NINF, 0, np.NINF, 0, 0, 0, 0, 0]
        L.x.upper_bounds = [np.PINF, 7,       1, 5,       1, 1, 1, 1, 1]

        U.c[U] = [2, 3, 4, 5, 6, 7]
        U.c[L] = [3, 4, 5, 6, 7, 8, 9, 10, 11]

        L.c[U] = [2, 3, 4, 5, 6, 7]
        L.c[L] = [3, 4, 5, 6, 7, 8, 9, 10, 11]

        U.A[U] = [[1, 2, 2, 3, 3, 3]]
        U.A[L] = [[1, 1, 2, 2, 2, 3, 3, 3, 3]]

        U.b = [2]

        L.A[U] = [[1, 2, 2, 3, 3, 3]]
        L.A[L] = [[1, 1, 2, 2, 2, 3, 3, 3, 3]]

        L.b = [3]

        lbp.check()

        self.assertEqual(len(U.x), 6)
        self.assertEqual(len(L.x), 9)

        self.assertEqual(len(U.c[U]), 6)
        self.assertEqual(len(U.c[L]), 9)

        self.assertEqual(len(L.c[U]), 6)
        self.assertEqual(len(L.c[L]), 9)

        convert_binaries_to_integers(lbp)
        lbp.check()
        
        self.assertEqual(len(U.x), 6)
        self.assertEqual(len(L.x), 9)

        self.assertEqual(len(U.c[U]), 6)
        #self.assertEqual(list(U.c[U]Z), [3,4,5,6,7])
        #self.assertEqual(U.c[U]B, None)
        self.assertEqual(len(U.c[L]), 9)
        #self.assertEqual(list(U.c[L]), [5,6,7,8,9,10,11])
        #self.assertEqual(U.c.L.xB, None)

        self.assertEqual(len(L.c[U]), 6)
        #self.assertEqual(list(L.c[U]Z), [3,4,5,6,7])
        #self.assertEqual(L.c[U]B, None)
        self.assertEqual(len(L.c[L]), 9)
        #self.assertEqual(list(L.c.L.xZ), [5,6,7,8,9,10,11])
        #self.assertEqual(L.c.L.xB, None)

        self.assertEqual(list(U.A[U].toarray()[0]), [1,2,2,3,3,3])
        #self.assertEqual(list(U.A[U]Z.toarray()[0]), [2,2,3,3,3])
        #self.assertEqual(U.A[U]B, None)
        self.assertEqual(list(U.A[L].toarray()[0]), [1,1,2,2,2,3,3,3,3])
        #self.assertEqual(list(U.A[L].toarray()[0]), [2,2,2,3,3,3,3])
        #self.assertEqual(U.A.L.xB, None)

        self.assertEqual(list(L.A[U].toarray()[0]), [1,2,2,3,3,3])
        #self.assertEqual(list(L.A[U]Z.toarray()[0]), [2,2,3,3,3])
        #self.assertEqual(L.A[U]B, None)
        self.assertEqual(list(L.A[L].toarray()[0]), [1,1,2,2,2,3,3,3,3])
        #self.assertEqual(list(L.A.L.xZ.toarray()[0]), [2,2,2,3,3,3,3])
        #self.assertEqual(L.A.L.xB, None)

    def test_test2(self):
        lbp = LinearMultilevelProblem()

        U = lbp.add_upper(nxR=1, nxZ=0, nxB=3)
        L = U.add_lower(nxR=2, nxZ=0, nxB=4)

        #U.xZ.lower_bounds    = [0, np.NINF]
        #U.xZ.upper_bounds    = [1, 5,     ]

        L.x.lower_bounds = [5,       np.NINF, 0, 0, 0, 0]
        L.x.upper_bounds = [np.PINF, 7      , 1, 1, 1, 1]
        #L.xZ.lower_bounds = [0, np.NINF, 0]
        #L.xZ.upper_bounds = [1, 5,       1]

        U.c[U] = [2, 5, 6, 7]
        U.c[L] = [3, 4, 8, 9, 10, 11]

        L.c[U] = [2, 5, 6, 7]
        L.c[L] = [3, 4, 8, 9, 10, 11]

        U.A[U] = [[1, 3, 3, 3]]
        U.A[L] = [[1, 1, 3, 3, 3, 3]]

        U.b = [2]

        L.A[U] = [[1, 3, 3, 3]]
        L.A[L] = [[1, 1, 3, 3, 3, 3]]

        L.b = [3]

        lbp.check()

        self.assertEqual(len(U.x), 4)
        self.assertEqual(len(L.x), 6)

        self.assertEqual(len(U.c[U]), 4)
        self.assertEqual(len(U.c[L]), 6)

        self.assertEqual(len(L.c[U]), 4)
        self.assertEqual(len(L.c[L]), 6)

        convert_binaries_to_integers(lbp)
        lbp.check()
        
        self.assertEqual(U.x.nxR, 1)
        self.assertEqual(U.x.nxZ, 3)
        self.assertEqual(U.x.nxB, 0)

        self.assertEqual(L.x.nxR, 2)
        self.assertEqual(L.x.nxZ, 4)
        self.assertEqual(L.x.nxB, 0)

        self.assertEqual(list(U.c[U]), [2,5,6,7])
        self.assertEqual(list(U.c[L]), [3,4,8,9,10,11])

        self.assertEqual(list(L.c[U]), [2,5,6,7])
        self.assertEqual(list(L.c[L]), [3,4,8,9,10,11])

        self.assertEqual(list(U.A[U].toarray()[0]), [1,3,3,3])
        self.assertEqual(list(U.A[L].toarray()[0]), [1,1,3,3,3,3])

        self.assertEqual(list(L.A[U].toarray()[0]), [1,3,3,3])
        self.assertEqual(list(L.A[L].toarray()[0]), [1,1,3,3,3,3])


class Test_Integers_Quadratic(Test_Integers_Linear):

    def _create(self):
        return QuadraticMultilevelProblem()


class Test_Examples_Linear(unittest.TestCase):

    def _create(self):
        return LinearMultilevelProblem()

    # NOTE - this is one of the few tests with binaries that have nonzero A-matrix coefficients
    def test_simple1(self):
        lbp = self._create()
        U = lbp.add_upper(nxR=4, nxZ=1, nxB=1)
        U.equalities = True
        U.x.lower_bounds = [np.NINF, -1,      np.NINF, -2, 0,       0]
        U.x.upper_bounds = [np.PINF, np.PINF, 5,        2, np.PINF, 1]
        U.c[U] = [3, -1, 0, 0, 1, 1]
        U.A[U] = [[-1, 6, -1, 1, -1, 0],
                  [ 0, 7,  0, 1,  0, 0],
                  [ 0, 0,  1, 1,  0, 1]]
        U.b = [-3, 5, 2]
        #lbp.print()
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.d, 1)
        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+2)
        self.assertEqual(list(ans.U.c[U]), [3, -1, 0, 0, -3, 0, 1, 1])
        self.assertEqual(ans.U.A[U].shape[0], lbp.U.A[U].shape[0]+1)
        self.assertEqual(ans.U.A[U].shape[1], lbp.U.A[U].shape[1]+2)

        self.assertEqual(len(ans.U.b), len(lbp.U.b)+1)
        self.assertEqual(ans.U.b[0], 10)
        self.assertEqual(ans.U.b[1], 14)
        self.assertEqual(ans.U.b[2], -1)
        self.assertEqual(ans.U.b[3], 4)
        self.assertEqual(len(ans.U.x), len(lbp.U.x)+2)


    # NOTE - An example where the upper-level doesn't have constraints with its own variables
    #        Q - Can this ever happen?
    def test_simple2(self):
        lbp = self._create()
        U = lbp.add_upper(nxB=1)
        L = U.add_lower(nxR=4, nxZ=1, nxB=1)

        U.inequalities = True
        U.A[L] = [[-1, 6, -1, 1, -1, 0],
                  [ 0, 7,  0, 1,  0, 0],
                  [ 0, 0,  1, 1,  0, 1]]
        U.b = [-3, 5, 2]

        L.inequalities = True
        L.x.lower_bounds = [np.NINF, -1,      np.NINF, -2, 0,       0]
        L.x.upper_bounds = [np.PINF, np.PINF, 5,        2, np.PINF, 1]

        L.c[U] = [-1]
        L.c[L] = [3, -1, 0, 0, 1, 1]

        L.A[U] = [[1], [2], [3]]
        L.A[L] = [[-1, 6, -1, 1, -1, 0],
                  [ 0, 7,  0, 1,  0, 0],
                  [ 0, 0,  1, 1,  0, 1]]
        L.b = [-3, 5, 2]
        #lbp.print()
        lbp.check()

        ans, soln_manager = convert_LinearMultilevelProblem_to_standard_form(lbp)
        #ans.print()
        ans.check()

        self.assertEqual(ans.U.LL.d, 1)
        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L])+5)
        self.assertEqual(list(ans.U.LL.c[L]), [3, -1, 0, 0, 0, 0, 0, -3, 0, 1, 1])
        self.assertEqual(ans.U.LL.A[U].shape[0], lbp.U.LL.A[U].shape[0]+1)
        self.assertEqual(ans.U.LL.A[U].shape[1], lbp.U.LL.A[U].shape[1]+3)
        self.assertEqual(ans.U.LL.A[L].shape[0], lbp.U.LL.A[L].shape[0]+1)
        self.assertEqual(ans.U.LL.A[L].shape[1], lbp.U.LL.A[L].shape[1]+5)

        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b)+1)
        self.assertEqual(ans.U.LL.b[0], 10)
        self.assertEqual(ans.U.LL.b[1], 14)
        self.assertEqual(ans.U.LL.b[2], -1)
        self.assertEqual(ans.U.LL.b[3], 4)

        self.assertEqual(len(ans.U.x), len(lbp.U.x)+3)
        self.assertEqual(len(ans.U.LL.x), len(lbp.U.LL.x)+5)


class Test_Examples_Quadratic(Test_Examples_Linear):

    def _create(self):
        return QuadraticMultilevelProblem()


if __name__ == "__main__":
    unittest.main()
