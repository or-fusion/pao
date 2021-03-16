import pprint
import numpy as np
import pyutilib.th as unittest
from pao.lbp import *
from pao.lbp.convert_repn import linearize_bilinear_terms


class Test_Trivial(unittest.TestCase):

    def _create(self):
        return QuadraticMultilevelProblem()

    def test_trivial1(self):
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=4)
        U.minimize = False
        U.c[U] = [1, 1, 1, 1, 1, 1]
        U.c[L] = [2, 2, 2, 2, 2, 2, 2]
        U.P[U,L] = (6,7), {(3,0):11, (4,1):13, (5,3):17}
        L.minimize = False
        L.c[U] = [3, 3, 3, 3, 3, 3]
        L.c[L] = [4, 4, 4, 4, 4, 4, 4]
        L.P[U,L] = (6,7), {(3,0):11, (4,1):13, (5,3):17}
        lbp.check()

        ans = linearize_bilinear_terms(lbp)
        ans.check()
        self.assertEqual(type(ans), LinearMultilevelProblem)

        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+3)
        self.assertEqual(len(ans.U.c[L]), len(lbp.U.c[L]))
        self.assertEqual(list(ans.U.c[U]), [1,0,0,0,1,1,1,1,1])
        self.assertEqual(list(ans.U.c[L]), [2,2,2,2,2,2,2])

        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+3)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L]))
        self.assertEqual(list(ans.U.LL.c[U]), [3,0,0,0,3,3,3,3,3])
        self.assertEqual(list(ans.U.LL.c[L]), [4,4,4,4,4,4,4])

        self.assertEqual(ans.U.d, 0)

        self.assertEqual(ans.U.A[U].toarray().tolist(),
 [[0.0, -1.0,  0.0,  0.0, 0.0, 0.0,  0.0,  0.0,  0.0],
  [0.0, -1.0,  0.0,  0.0, 0.0, 0.0,  1.0,  0.0,  0.0],
  [0.0,  1.0,  0.0,  0.0, 0.0, 0.0,  0.0,  0.0,  0.0],
  [0.0,  1.0,  0.0,  0.0, 0.0, 0.0, -1.0,  0.0,  0.0],
  [0.0,  0.0, -1.0,  0.0, 0.0, 0.0,  0.0,  0.0,  0.0],
  [0.0,  0.0, -1.0,  0.0, 0.0, 0.0,  0.0,  1.0,  0.0],
  [0.0,  0.0,  1.0,  0.0, 0.0, 0.0,  0.0,  0.0,  0.0],
  [0.0,  0.0,  1.0,  0.0, 0.0, 0.0,  0.0, -1.0,  0.0],
  [0.0,  0.0,  0.0, -1.0, 0.0, 0.0,  0.0,  0.0,  0.0],
  [0.0,  0.0,  0.0, -1.0, 0.0, 0.0,  0.0,  0.0,  1.0],
  [0.0,  0.0,  0.0,  1.0, 0.0, 0.0,  0.0,  0.0,  0.0],
  [0.0,  0.0,  0.0,  1.0, 0.0, 0.0,  0.0,  0.0, -1.0]])
 
        self.assertEqual(ans.U.A[L].toarray().tolist(),
[[-1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [ 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [-1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertEqual(len(ans.U.b), len(lbp.U.b)+12)

    def test_trivial2(self):
        # Adding more bilinear terms
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=4)
        U.minimize = False
        U.c[U] = [1, 1, 1, 1, 1, 1]
        U.c[L] = [2, 2, 2, 2, 2, 2, 2]
        U.P[U,L] = (6,7), {(3,0):11, (4,1):13, (5,3):17}
        L.minimize = False
        L.c[U] = [3, 3, 3, 3, 3, 3]
        L.c[L] = [4, 4, 4, 4, 4, 4, 4]
        L.P[U,L] = (6,7), {(3,1):11, (4,2):13, (5,4):17}
        lbp.check()

        ans = linearize_bilinear_terms(lbp)
        ans.check()
        self.assertEqual(type(ans), LinearMultilevelProblem)

        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+6)
        self.assertEqual(len(ans.U.c[L]), len(lbp.U.c[L]))
        self.assertEqual(list(ans.U.c[U]), [1,0,0,0,0,0,0,1,1,1,1,1])
        self.assertEqual(list(ans.U.c[L]), [2,2,2,2,2,2,2])

        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L]))
        self.assertEqual(list(ans.U.LL.c[U]), [3,0,0,0,0,0,0,3,3,3,3,3])
        self.assertEqual(list(ans.U.LL.c[L]), [4,4,4,4,4,4,4])

        self.assertEqual(ans.U.d, 0)

        #pprint.pprint(ans.U.A[U].toarray().tolist())
        self.assertEqual(ans.U.A[U].toarray().tolist(),
[[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]])

        #pprint.pprint(ans.U.A[L].toarray().tolist())
        self.assertEqual(ans.U.A[L].toarray().tolist(),
[[-1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [-1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -1000000.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertEqual(len(ans.U.b), len(lbp.U.b)+24)
        #pprint.pprint(list(ans.U.b))
        self.assertEqual(list(ans.U.b), 
[0.0,
 1000000.0,
 0.0,
 1000000.0,
 0.0,
 1000000.0,
 0.0,
 1000000.0,
 0.0,
 1.0,
 0.0,
 -0.0,
 0.0,
 1000000.0,
 0.0,
 1000000.0,
 0.0,
 1000000.0,
 0.0,
 1000000.0,
 0.0,
 1.0,
 0.0,
 -0.0])

    def test_trivial3(self):
        # Adding more bilinear terms
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=4)
        U.x.lower_bounds = [11,11,11,0,0,0]
        U.x.upper_bounds = [13,13,13,1,1,1]
        L.x.lower_bounds = [15,15,15,0,0,0,0]
        L.x.upper_bounds = [17,17,17,1,1,1,1]
        U.minimize = False
        U.c[U] = [1, 1, 1, 1, 1, 1]
        U.c[L] = [2, 2, 2, 2, 2, 2, 2]
        U.P[U,L] = (6,7), {(3,0):11, (4,1):13, (5,3):17}
        L.minimize = False
        L.c[U] = [3, 3, 3, 3, 3, 3]
        L.c[L] = [4, 4, 4, 4, 4, 4, 4]
        L.P[U,L] = (6,7), {(3,1):11, (4,2):13, (5,4):17}
        lbp.check()

        ans = linearize_bilinear_terms(lbp)
        ans.check()
        self.assertEqual(type(ans), LinearMultilevelProblem)

        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+6)
        self.assertEqual(len(ans.U.c[L]), len(lbp.U.c[L]))
        self.assertEqual(list(ans.U.c[U]), [1,0,0,0,0,0,0,1,1,1,1,1])
        self.assertEqual(list(ans.U.c[L]), [2,2,2,2,2,2,2])

        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L]))
        self.assertEqual(list(ans.U.LL.c[U]), [3,0,0,0,0,0,0,3,3,3,3,3])
        self.assertEqual(list(ans.U.LL.c[L]), [4,4,4,4,4,4,4])

        self.assertEqual(ans.U.d, 0)

        #pprint.pprint(ans.U.A[U].toarray().tolist())
        self.assertEqual(ans.U.A[U].toarray().tolist(),
[[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]])

        #pprint.pprint(ans.U.A[L].toarray().tolist())
        self.assertEqual(ans.U.A[L].toarray().tolist(),
[[15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [-17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [-15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -17.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertEqual(len(ans.U.b), len(lbp.U.b)+24)
        self.assertEqual(list(ans.U.b), 
[0.0,
17.0,
0.0,
-15.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
1.0,
0.0,
-0.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
1.0,
0.0,
-0.0])

    def test_trivial4(self):
        # Adding more bilinear terms
        lbp = self._create()
        U = lbp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = U.add_lower(nxR=1, nxZ=2, nxB=4)
        U.x.lower_bounds = [11,11,11,0,0,0]
        U.x.upper_bounds = [13,13,13,1,1,1]
        L.x.lower_bounds = [15,15,15,0,0,0,0]
        L.x.upper_bounds = [17,17,17,1,1,1,1]
        U.minimize = False
        U.c[U] = [1, 1, 1, 1, 1, 1]
        U.c[L] = [2, 2, 2, 2, 2, 2, 2]
        U.P[U,L] = (6,7), {(3,0):11, (4,1):13, (5,3):17}
        U.A[U] = (1,6),{(0,5):2}
        U.A[L] = (1,7),{(0,6):2}
        U.b = [2]
        L.minimize = False
        L.c[U] = [3, 3, 3, 3, 3, 3]
        L.c[L] = [4, 4, 4, 4, 4, 4, 4]
        L.P[U,L] = (6,7), {(3,1):11, (4,2):13, (5,4):17}
        L.A[U] = (1,6),{(0,5):-2}
        L.A[L] = (1,7),{(0,6):-2}
        L.b = [-2]
        lbp.check()

        ans = linearize_bilinear_terms(lbp)
        ans.check()
        self.assertEqual(type(ans), LinearMultilevelProblem)

        self.assertEqual(len(ans.U.c[U]), len(lbp.U.c[U])+6)
        self.assertEqual(len(ans.U.c[L]), len(lbp.U.c[L]))
        self.assertEqual(list(ans.U.c[U]), [1,0,0,0,0,0,0,1,1,1,1,1])
        self.assertEqual(list(ans.U.c[L]), [2,2,2,2,2,2,2])

        self.assertEqual(len(ans.U.LL.c[U]), len(lbp.U.LL.c[U])+6)
        self.assertEqual(len(ans.U.LL.c[L]), len(lbp.U.LL.c[L]))
        self.assertEqual(list(ans.U.LL.c[U]), [3,0,0,0,0,0,0,3,3,3,3,3])
        self.assertEqual(list(ans.U.LL.c[L]), [4,4,4,4,4,4,4])

        self.assertEqual(ans.U.d, 0)

        #pprint.pprint(ans.U.A[U].toarray().tolist())
        self.assertEqual(ans.U.A[U].toarray().tolist(),
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]])

        #pprint.pprint(ans.U.A[L].toarray().tolist())
        self.assertEqual(ans.U.A[L].toarray().tolist(),
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
 [15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [-17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [-15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -17.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -17.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertEqual(len(ans.U.b), len(lbp.U.b)+24)
        self.assertEqual(list(ans.U.b), 
[2.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
1.0,
0.0,
-0.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
17.0,
0.0,
-15.0,
0.0,
1.0,
0.0,
-0.0])

        #pprint.pprint(ans.U.LL.A[U].toarray().tolist())
        self.assertEqual(ans.U.LL.A[U].toarray().tolist(),
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0]])

        #pprint.pprint(ans.U.LL.A[L].toarray().tolist())
        self.assertEqual(ans.U.LL.A[L].toarray().tolist(),
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0]])

        self.assertEqual(len(ans.U.LL.b), len(lbp.U.LL.b))
        self.assertEqual(list(ans.U.LL.b), [-2])


if __name__ == "__main__":
    unittest.main()
