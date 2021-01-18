import numpy as np
import scipy.sparse
import pyutilib.th as unittest
from pao.lbp import *
from pao.lbp.repn import SimplifiedList, LinearLevelRepn, LevelVariable, LevelValues, LevelValueWrapper


class A(object):

    def __init__(self, i=0):
        self.i = i

    def clone(self, parent=None):
        return A(self.i)


class Test_SimplifiedList(unittest.TestCase):

    def test_init(self):
        l = SimplifiedList()
        self.assertEqual(len(l._data),0)

    def test_clone1(self):
        l = SimplifiedList()
        a0 = A()
        a1 = A(1)
        a1.clone()
        l.append(a0)
        l.append(a1)
        ans = l.clone()
        self.assertEqual(len(ans), 2)
        
    def test_clone2(self):
        l = SimplifiedList()
        a0 = A()
        a1 = A(1)
        l.append(a0)
        l.append(a1)
        self.assertEqual(len(l), 2)
        self.assertEqual(l[0].i, 0)
        self.assertEqual(l[1].i, 1)

        ans = l.clone()
        self.assertEqual(len(ans), 2)
        self.assertEqual(ans[0].i, 0)
        self.assertEqual(ans[1].i, 1)
        self.assertEqual(ans[-1].i, 1)
        
    def test_insert(self):
        l = SimplifiedList()
        l.append(A())
        self.assertEqual(len(l), 1)
        l.insert(2, A(2))
        self.assertEqual(len(l), 2)
        
    def test_iter(self):
        l = SimplifiedList()
        a0 = A(0)
        l.append(a0)
        a2 = A(2)
        l.append(a2)
        a1 = A(1)
        l.append(a1)
        self.assertEqual([v.i for v in l], [0,2,1])

    def test_getitem(self):
        l = SimplifiedList()
        a = A()
        l.append(a)
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].i, 0)
        self.assertEqual(len(l), 1)
        try:
            l[2].i
            self.fail("Expected IndexError")
        except IndexError:
            pass
        
    def test_setitem(self):
        l = SimplifiedList()
        a = A()
        l.append(a)
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].i, 0)
        self.assertEqual(len(l), 1)

        try:
            l[3] = A(3)
            self.fail("Expected IndexError")
        except IndexError:
            pass

    def test_delitem(self):
        l = SimplifiedList()
        l.append(A(1))
        l.append(A(2))
        l.append(A(3))

        del l[1]
        self.assertEqual(len(l), 2)

        try:
            del l[2]
            self.fail("Expected IndexError")
        except IndexError:
            pass

    def test_len(self):
        l = SimplifiedList()
        a1 = A(1)
        a2 = A(2)
        l.append(a1)
        l.append(a2)
        self.assertEqual(len(l), len(l._data))

    def test_getattr(self):
        l = SimplifiedList()
        try:
            l.i
            self.fail("Expected AssertionError")
        except:
            pass
       
        a1 = A(1)
        l.append(a1)
        x = l.i
        self.assertEqual(l.i, 1) 

        a2 = A(2)
        l.append(a2)
        try:
            l.i
            self.fail("Expected AssertionError")
        except:
            pass

        self.assertEqual( getattr(l, '_foo', None), None )

    def test_setattr(self):
        l = SimplifiedList()
        try:
            l.i = 0
            self.fail("Expected AssertionError")
        except:
            pass

        a1 = A(1)
        l.append(a1)
        l.i = 0

        l.append(a1)
        try:
            l.i = 0
            self.fail("Expected AssertionError")
        except:
            pass


class Test_LevelVariable(unittest.TestCase):

    def test_init(self):
        l = LevelVariable(10)
        self.assertEqual(len(l.values), 10)
        self.assertEqual(l.num, 10)

    def test_init_bounds(self):
        l = LevelVariable(3, lb=[1,2,3], ub=[5,6,7])
        self.assertEqual(len(l.values), 3)
        self.assertEqual(l.num, 3)
        self.assertEqual(len(l.lower_bounds), 3)
        self.assertEqual(len(l.upper_bounds), 3)

    def test_len(self):
        l = LevelVariable(10)
        self.assertEqual(len(l), 10)

    def test_iter(self):
        l = LevelVariable(10)
        tmp = [i for i in l]
        self.assertEqual(tmp, list(range(10)))

    def test_setattr(self):
        l = LevelVariable(3)
        l.a = [1,2,3]
        self.assertEqual(type(l.a),list)
        l.lower_bounds = [1,2,3]
        self.assertEqual(type(l.lower_bounds),np.ndarray)

    def test_clone(self):
        l = LevelVariable(3)
        l.lower_bounds = [1,2,3]
        ans = l.clone()
        self.assertEqual(len(ans), len(l))
        self.assertEqual(type(l.lower_bounds),np.ndarray)

    def test_resize(self):
        l = LevelVariable(5)
        l.lower_bounds = [1,2,3,np.NINF,np.NINF]
        self.assertEqual(list(l.lower_bounds), [1,2,3,np.NINF,np.NINF])

        l = LevelVariable(3,0,0)
        l.lower_bounds = [1,2,3]
        l._resize(3, 0, 0, lb=np.NINF)
        self.assertEqual(list(l.lower_bounds), [1,2,3])

        l = LevelVariable(3,0,0)
        l.lower_bounds = [1,2,3]
        l.upper_bounds = [4,5,6]
        l._resize(2,0,0, lb=4)
        self.assertEqual(list(l.lower_bounds), [1,2])
        self.assertEqual(list(l.upper_bounds), [4,5])

        l = LevelVariable(3,0,0)
        l.lower_bounds = [1,2,3]
        l.upper_bounds = [3,4,5]
        l._resize(4,0,0, lb=4, ub=6)
        self.assertEqual(list(l.lower_bounds), [1,2,3,4])
        self.assertEqual(list(l.upper_bounds), [3,4,5,6])

        l = LevelVariable(3,0,0)
        l._resize(4,0,0, lb=4)
        self.assertEqual(list(l.lower_bounds), [np.NINF,np.NINF,np.NINF,4])
        self.assertEqual(list(l.upper_bounds), [np.PINF,np.PINF,np.PINF,np.PINF])

        l = LevelVariable(3,4,5)
        l._resize(2,3,4, lb=4)
        self.assertEqual(list(l.lower_bounds), [np.NINF,np.NINF,np.NINF,np.NINF,np.NINF,0,0,0,0])
        self.assertEqual(list(l.upper_bounds), [np.PINF,np.PINF,np.PINF,np.PINF,np.PINF,1,1,1,1])


class Test_LevelValues(unittest.TestCase):

    def test_init(self):
        l = LevelValues()
        self.assertEqual(l._matrix, False)
        self.assertEqual(l.x, None)

    def test_init_matrix(self):
        l = LevelValues(matrix=True)
        self.assertEqual(l._matrix, True)
        self.assertEqual(l.x, None)

    def test_set_values(self):
        l = LevelValues(matrix=True)
        l.set_values(x=1)
        self.assertEqual(l.x, 1)

    def test_len(self):
        l = LevelValues()
        l.set_values(x=[0,1,2,3,4,5])
        self.assertEqual(len(l), 6)
        
    def test_len_matrix(self):
        l = LevelValues(matrix=True)
        l.set_values(x=[[0,2,0,0,0,0,0,5],[0,0,0,0,0,0,1,6],[0,0,0,0,5,0,0,0]])
        self.assertEqual(len(l), 3)
        
    def test_clone_values(self):
        l = LevelValues()
        l.set_values(x=[0,1,2,3,4,5])
        ans = l.clone()
        self.assertEqual(len(ans), 6)
        
    def test_clone_matrix(self):
        l = LevelValues(matrix=True)
        l.set_values(x=[[0,2,0,0,0,0,0,5],[0,0,0,0,0,0,1,6],[0,0,0,0,5,0,0,0]])
        ans = l.clone()
        self.assertEqual(len(ans), 3)
        
    def test_setattr(self):
        l = LevelValues()
        l.a = [1,2,3]
        self.assertEqual(type(l.a),list)
        l.x = [1,2,3]
        self.assertEqual(type(l.x),np.ndarray)

    def test_setattr_matrix(self):
        l = LevelValues(matrix=True)
        l.a = [[1,0,2,0],
               [0,3,0,4]]
        self.assertEqual(type(l.a),list)
        l.x = [[1,0,2,0],
                [0,3,0,4]]
        self.assertEqual(type(l.x),scipy.sparse.csr.csr_matrix)


class Test_LevelValueWrapper(unittest.TestCase):

    def test_init(self):
        l = LevelValueWrapper('foo')
        self.assertEqual(l._matrix, False)
        self.assertEqual(l._values, {})
        self.assertEqual(l._prefix, 'foo')

    def test_init_matrix(self):
        l = LevelValueWrapper('foo', matrix=True)
        self.assertEqual(l._matrix, True)
        self.assertEqual(l._values, {})
        self.assertEqual(l._prefix, 'foo')

    def test_len(self):
        l = LevelValueWrapper('foo', matrix=True)
        self.assertEqual(len(l), 0)
        L0 = LinearLevelRepn(1,2,3)
        L1 = LinearLevelRepn(1,2,3)
        l[L0] = [1,2,3]
        l[L1] = [1,2,3]
        self.assertEqual(len(l), 2)

    def test_iter(self):
        l = LevelValueWrapper('foo', matrix=True)
        self.assertEqual(len(l), 0)
        L0 = LinearLevelRepn(1,2,3)
        L1 = LinearLevelRepn(1,2,3)
        L2 = LinearLevelRepn(1,2,3)
        l[L0] = [1,2,3]
        l[L1] = [1,2,3]
        l[L2] = [1,2,3]
        self.assertEqual(len(l), 3)
        self.assertEqual(sorted(i for i in l), sorted([L0.id, L1.id, L2.id]))

    def test_getattr(self):
        l = LevelValueWrapper('foo', matrix=True)
        try:
            l.foo
        except AttributeError:
            pass
        try:
            l.L
        except AttributeError:
            pass
        
    def test_setgetitem(self):
        l = LevelValueWrapper('foo', matrix=False)
        L0 = LinearLevelRepn(1,2,3)
        L1 = LinearLevelRepn(1,2,3)
        l[L0] = [1,2,3]
        l[L1.id] = [4,5,6]
        self.assertEqual(len(l), 2)
        self.assertEqual(list(l[L0.id]), [1,2,3])
        self.assertEqual(list(l[L1]), [4,5,6])

    def test_clone(self):
        l = LevelValueWrapper('foo', matrix=True)
        try:
            l.foo
        except AttributeError:
            pass
        try:
            l.L
        except AttributeError:
            pass
        ans = l.clone()
        self.assertEqual(list(ans._values.keys()), [])
        try:
            ans.foo
        except AttributeError:
            pass
        try:
            ans.L
        except AttributeError:
            pass
        

class Test_LinearLevelRepn(unittest.TestCase):

    def test_init(self):
        l = LinearLevelRepn(1,2,3)
        self.assertEqual(len(l.x), 6)
        self.assertEqual(l.minimize, True)
        self.assertEqual(len(l.c), 0)
        self.assertEqual(l.d, 0)
        self.assertEqual(len(l.A), 0)
        self.assertEqual(l.b.size, 0)
        self.assertEqual(l.inequalities, True)
        self.assertEqual(l.equalities, False)

    def test_inequalities(self):
        l = LinearLevelRepn(1,2,3)
        self.assertEqual(l.inequalities, True)
        self.assertEqual(l.equalities, False)
        l.inequalities=False
        self.assertEqual(l.inequalities, False)
        self.assertEqual(l.equalities, True)
        l.equalities=False
        self.assertEqual(l.inequalities, True)
        self.assertEqual(l.equalities, False)

    def test_setattr(self):
        l = LinearLevelRepn(1,2,3)
        l.y = -1
        self.assertEqual(l.y, -1)
        l.b = -1
        self.assertEqual(l.b, np.array(-1))

    def test_clone(self):
        l = LinearLevelRepn(1,2,3)
        l.y = -1
        l.b = -1
        ans = l.clone()
        self.assertEqual(ans.y, -1)
        self.assertEqual(ans.b, np.array(-1))
        

class Test_LinearMultilevelProblem(unittest.TestCase):

    def test_init(self):
        blp = LinearMultilevelProblem()
        self.assertEqual(blp.name, None)
        
    def test_init_name(self):
        blp = LinearMultilevelProblem('foo')
        self.assertEqual(blp.name, 'foo')

    def test_names(self):
        blp = LinearMultilevelProblem()
        A = blp.add_upper(nxR=1, nxZ=2, nxB=3, name='A')
        B = A.add_lower(nxR=1, nxZ=2, nxB=3, name='B')
        C = A.add_lower(nxR=1, nxZ=2, nxB=3, name='C')
        self.assertEqual(A.name, 'A')
        self.assertEqual(B.name, 'B')
        self.assertEqual(C.name, 'C')

    def test_levels(self):
        blp = LinearMultilevelProblem()
        A = blp.add_upper(nxR=1, nxZ=2, nxB=3, name='A', id=100)
        B = A.add_lower(nxR=1, nxZ=2, nxB=3, name='B', id=-1)
        C = A.add_lower(nxR=1, nxZ=2, nxB=3, name='C', id=2)
        D = B.add_lower(nxR=1, nxZ=2, nxB=3, name='D', id=10)
        E = B.add_lower(nxR=1, nxZ=2, nxB=3, name='E', id=11)
        self.assertEqual(A.id, 100)
        names = [L.name for L in blp.levels()]

        self.assertEqual(names, ['A', 'B', 'D', 'E', 'C'])
        self.assertEqual(A.UL(), None)
        self.assertEqual(B.UL().id, 100)
        self.assertEqual(C.UL().id, 100)
        self.assertEqual(D.UL().id, -1)
        self.assertEqual(E.UL().id, -1)

    def test_add_upper(self):
        blp = LinearMultilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        self.assertEqual(len(U.x), 6)
        self.assertEqual(id(U), id(blp.U))

    def test_add_lower(self):
        blp = LinearMultilevelProblem()
        U = blp.add_upper(nxR=1)
        L = []
        L.append( U.add_lower(nxR=1, nxZ=2, nxB=3) )
        self.assertEqual(len(L[0].x), 6)
        self.assertEqual(id(L[0]), id(blp.U.LL[0]))

        L.append( blp.U.add_lower(nxR=1, nxZ=2, nxB=3) )
        self.assertEqual(len(L[1].x), 6)
        self.assertEqual(id(L[0]), id(blp.U.LL[0]))
        self.assertEqual(id(L[1]), id(blp.U.LL[1]))

    def test_clone(self):
        blp = LinearMultilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        U.add_lower(nxR=1, nxZ=2, nxB=4)
        U.add_lower(nxR=1, nxZ=2, nxB=5)

        U.c[U] = [0]*6
        U.A[U] = [[1]*6]
        U.b = [0]

        ans = blp.clone()

        self.assertEqual(len(ans.U.x), 6)
        self.assertEqual(len(ans.U.c[U]), 6)
        self.assertEqual(len(ans.U.c[U]), 6)
        self.assertEqual(len(U.LL[0].x), 7)
        self.assertEqual(len(U.LL[1].x), 8)

    def test_resize(self):
        def tmp():
            blp = LinearMultilevelProblem()
            U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
            L0 = U.add_lower(nxR=1, nxZ=2, nxB=4, name='L0')
            L1 = U.add_lower(nxR=1, nxZ=2, nxB=5, name='L1')
            L2 = L0.add_lower(nxR=1, nxZ=3, nxB=4, name='L2')
            L3 = L0.add_lower(nxR=1, nxZ=4, nxB=4, name='L3')
            L4 = L1.add_lower(nxR=2, nxZ=2, nxB=4, name='L4')
            L5 = L1.add_lower(nxR=3, nxZ=2, nxB=4, name='L5')

            U.c[U] = [-1]*6
            U.A[U] = [[1]*6]
            U.c[L0] = [2]*7
            U.A[L0] = [[3]*7]
            U.c[L1] = [2]*8
            U.A[L1] = [[3]*8]
            U.c[L2] = [4]*8
            U.A[L2] = [[5]*8]
            U.c[L3] = [6]*9
            U.A[L3] = [[7]*9]
            U.c[L4] = [8]*8
            U.A[L4] = [[9]*8]
            U.c[L5] = [10]*9
            U.A[L5] = [[11]*9]
            U.b = [0]

            L0.c[U] = [-1]*6
            L0.A[U] = [[1]*6]
            L0.c[L0] = [2]*7
            L0.A[L0] = [[3]*7]
            L0.c[L2] = [4]*8
            L0.A[L2] = [[5]*8]
            L0.c[L3] = [6]*9
            L0.A[L3] = [[7]*9]
            U.b = [1]

            L2.c[U] = [-1]*6
            L2.A[U] = [[1]*6]
            L2.c[L0] = [2]*7
            L2.A[L0] = [[3]*7]
            L2.c[L2] = [4]*8
            L2.A[L2] = [[5]*8]
            U.b = [2]

            L3.c[U] = [-1]*6
            L3.A[U] = [[1]*6]
            L3.c[L0] = [2]*7
            L3.A[L0] = [[3]*7]
            L3.c[L3] = [6]*9
            L3.A[L3] = [[7]*9]
            U.b = [3]

            L1.c[U] = [-1]*6
            L1.A[U] = [[1]*6]
            L1.c[L1] = [2]*8
            L1.A[L1] = [[3]*8]
            L1.c[L4] = [8]*8
            L1.A[L4] = [[9]*8]
            L1.c[L5] = [10]*9
            L1.A[L5] = [[11]*9]
            U.b = [4]

            L4.c[U] = [-1]*6
            L4.A[U] = [[1]*6]
            L4.c[L1] = [2]*8
            L4.A[L1] = [[3]*8]
            L4.c[L4] = [8]*8
            L4.A[L4] = [[9]*8]
            U.b = [5]

            L5.c[U] = [-1]*6
            L5.A[U] = [[1]*6]
            L5.c[L1] = [2]*8
            L5.A[L1] = [[3]*8]
            L5.c[L5] = [10]*9
            L5.A[L5] = [[11]*9]
            U.b = [6]

            return blp, U, L0, L1, L2, L3, L4, L5

        #
        # U.resize
        #
        blp, U, L0, L1, L2, L3, L4, L5 = tmp()
        U.resize(nxR=2, nxZ=3, nxB=4)
        self.assertEqual(list(U.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])
        self.assertEqual(list(L0.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])
        self.assertEqual(list(L1.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])
        self.assertEqual(list(L2.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])
        self.assertEqual(list(L3.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])
        self.assertEqual(list(L4.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])
        self.assertEqual(list(L5.c[U]), [-1, 0, -1, -1, 0, -1, -1, -1, 0])

        self.assertEqual([U.A[U].todok()[0,i] for i in range(U.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])
        self.assertEqual([L0.A[U].todok()[0,i] for i in range(L0.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])
        self.assertEqual([L1.A[U].todok()[0,i] for i in range(L1.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])
        self.assertEqual([L2.A[U].todok()[0,i] for i in range(L2.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])
        self.assertEqual([L3.A[U].todok()[0,i] for i in range(L3.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])
        self.assertEqual([L4.A[U].todok()[0,i] for i in range(L4.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])
        self.assertEqual([L5.A[U].todok()[0,i] for i in range(L5.A[U].shape[1])], [1,0,1,1,0,1,1,1,0])

        self.assertEqual(list(U.c[L0]), [2]*7)
        self.assertEqual(list(U.c[L1]), [2]*8)
        self.assertEqual(list(U.c[L2]), [4]*8)
        self.assertEqual(list(U.c[L3]), [6]*9)
        self.assertEqual(list(U.c[L4]), [8]*8)
        self.assertEqual(list(U.c[L5]), [10]*9)

        self.assertEqual([U.A[L0].todok()[0,i] for i in range(U.A[L0].shape[1])], [3]*7)
        self.assertEqual([U.A[L1].todok()[0,i] for i in range(U.A[L1].shape[1])], [3]*8)
        self.assertEqual([U.A[L2].todok()[0,i] for i in range(U.A[L2].shape[1])], [5]*8)
        self.assertEqual([U.A[L3].todok()[0,i] for i in range(U.A[L3].shape[1])], [7]*9)
        self.assertEqual([U.A[L4].todok()[0,i] for i in range(U.A[L4].shape[1])], [9]*8)
        self.assertEqual([U.A[L5].todok()[0,i] for i in range(U.A[L5].shape[1])], [11]*9)

        self.assertEqual(list(L0.c[L0]), [2]*7)
        self.assertEqual(list(L0.c[L2]), [4]*8)
        self.assertEqual(list(L0.c[L3]), [6]*9)

        self.assertEqual([L0.A[L0].todok()[0,i] for i in range(L0.A[L0].shape[1])], [3]*7)
        self.assertEqual([L0.A[L2].todok()[0,i] for i in range(L0.A[L2].shape[1])], [5]*8)
        self.assertEqual([L0.A[L3].todok()[0,i] for i in range(L0.A[L3].shape[1])], [7]*9)

        self.assertEqual(list(L1.c[L1]), [2]*8)
        self.assertEqual(list(L1.c[L4]), [8]*8)
        self.assertEqual(list(L1.c[L5]), [10]*9)

        self.assertEqual([L1.A[L1].todok()[0,i] for i in range(L1.A[L1].shape[1])], [3]*8)
        self.assertEqual([L1.A[L4].todok()[0,i] for i in range(L1.A[L4].shape[1])], [9]*8)
        self.assertEqual([L1.A[L5].todok()[0,i] for i in range(L1.A[L5].shape[1])], [11]*9)

        self.assertEqual(list(L2.c[L0]), [2]*7)
        self.assertEqual(list(L2.c[L2]), [4]*8)

        self.assertEqual([L2.A[L0].todok()[0,i] for i in range(L2.A[L0].shape[1])], [3]*7)
        self.assertEqual([L2.A[L2].todok()[0,i] for i in range(L2.A[L2].shape[1])], [5]*8)

        self.assertEqual(list(L3.c[L0]), [2]*7)
        self.assertEqual(list(L3.c[L3]), [6]*9)

        self.assertEqual([L3.A[L0].todok()[0,i] for i in range(L3.A[L0].shape[1])], [3]*7)
        self.assertEqual([L3.A[L3].todok()[0,i] for i in range(L3.A[L3].shape[1])], [7]*9)

        self.assertEqual(list(L4.c[L1]), [2]*8)
        self.assertEqual(list(L4.c[L4]), [8]*8)

        self.assertEqual([L4.A[L1].todok()[0,i] for i in range(L4.A[L1].shape[1])], [3]*8)
        self.assertEqual([L4.A[L4].todok()[0,i] for i in range(L4.A[L4].shape[1])], [9]*8)

        self.assertEqual(list(L5.c[L1]), [2]*8)
        self.assertEqual(list(L5.c[L5]), [10]*9)

        self.assertEqual([L5.A[L1].todok()[0,i] for i in range(L5.A[L1].shape[1])], [3]*8)
        self.assertEqual([L5.A[L5].todok()[0,i] for i in range(L5.A[L5].shape[1])], [11]*9)

        #
        # L0.resize
        #
        blp, U, L0, L1, L2, L3, L4, L5 = tmp()
        L0.resize(nxR=2, nxZ=3, nxB=4)
        self.assertEqual(list(U.c[U]),  [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L0.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L1.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L2.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L3.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L4.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L5.c[U]), [-1, -1, -1, -1, -1, -1])

        self.assertEqual([U.A[U].todok()[0,i] for i in range(U.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L0.A[U].todok()[0,i] for i in range(L0.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L1.A[U].todok()[0,i] for i in range(L1.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L2.A[U].todok()[0,i] for i in range(L2.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L3.A[U].todok()[0,i] for i in range(L3.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L4.A[U].todok()[0,i] for i in range(L4.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L5.A[U].todok()[0,i] for i in range(L5.A[U].shape[1])], [1,1,1,1,1,1])

        self.assertEqual(list(U.c[L0]),  [2,0,2,2,0,2,2,2,2])
        self.assertEqual(list(L0.c[L0]), [2,0,2,2,0,2,2,2,2])
        self.assertEqual(list(L2.c[L0]), [2,0,2,2,0,2,2,2,2])
        self.assertEqual(list(L3.c[L0]), [2,0,2,2,0,2,2,2,2])

        self.assertEqual([U.A[L0].todok()[0,i] for i in range(U.A[L0].shape[1])],   [3,0,3,3,0,3,3,3,3])
        self.assertEqual([L0.A[L0].todok()[0,i] for i in range(L0.A[L0].shape[1])], [3,0,3,3,0,3,3,3,3])
        self.assertEqual([L2.A[L0].todok()[0,i] for i in range(L2.A[L0].shape[1])], [3,0,3,3,0,3,3,3,3])
        self.assertEqual([L3.A[L0].todok()[0,i] for i in range(L3.A[L0].shape[1])], [3,0,3,3,0,3,3,3,3])

        self.assertEqual(list(U.c[L1]), [2]*8)
        self.assertEqual(list(U.c[L2]), [4]*8)
        self.assertEqual(list(U.c[L3]), [6]*9)
        self.assertEqual(list(U.c[L4]), [8]*8)
        self.assertEqual(list(U.c[L5]), [10]*9)

        self.assertEqual([U.A[L1].todok()[0,i] for i in range(U.A[L1].shape[1])], [3]*8)
        self.assertEqual([U.A[L2].todok()[0,i] for i in range(U.A[L2].shape[1])], [5]*8)
        self.assertEqual([U.A[L3].todok()[0,i] for i in range(U.A[L3].shape[1])], [7]*9)
        self.assertEqual([U.A[L4].todok()[0,i] for i in range(U.A[L4].shape[1])], [9]*8)
        self.assertEqual([U.A[L5].todok()[0,i] for i in range(U.A[L5].shape[1])], [11]*9)

        self.assertEqual(list(L0.c[L2]), [4]*8)
        self.assertEqual(list(L0.c[L3]), [6]*9)

        self.assertEqual([L0.A[L2].todok()[0,i] for i in range(L0.A[L2].shape[1])], [5]*8)
        self.assertEqual([L0.A[L3].todok()[0,i] for i in range(L0.A[L3].shape[1])], [7]*9)

        self.assertEqual(list(L1.c[L1]), [2]*8)
        self.assertEqual(list(L1.c[L4]), [8]*8)
        self.assertEqual(list(L1.c[L5]), [10]*9)

        self.assertEqual([L1.A[L1].todok()[0,i] for i in range(L1.A[L1].shape[1])], [3]*8)
        self.assertEqual([L1.A[L4].todok()[0,i] for i in range(L1.A[L4].shape[1])], [9]*8)
        self.assertEqual([L1.A[L5].todok()[0,i] for i in range(L1.A[L5].shape[1])], [11]*9)

        self.assertEqual(list(L2.c[L2]), [4]*8)

        self.assertEqual([L2.A[L2].todok()[0,i] for i in range(L2.A[L2].shape[1])], [5]*8)

        self.assertEqual(list(L3.c[L3]), [6]*9)

        self.assertEqual([L3.A[L3].todok()[0,i] for i in range(L3.A[L3].shape[1])], [7]*9)

        self.assertEqual(list(L4.c[L1]), [2]*8)
        self.assertEqual(list(L4.c[L4]), [8]*8)

        self.assertEqual([L4.A[L1].todok()[0,i] for i in range(L4.A[L1].shape[1])], [3]*8)
        self.assertEqual([L4.A[L4].todok()[0,i] for i in range(L4.A[L4].shape[1])], [9]*8)

        self.assertEqual(list(L5.c[L1]), [2]*8)
        self.assertEqual(list(L5.c[L5]), [10]*9)

        self.assertEqual([L5.A[L1].todok()[0,i] for i in range(L5.A[L1].shape[1])], [3]*8)
        self.assertEqual([L5.A[L5].todok()[0,i] for i in range(L5.A[L5].shape[1])], [11]*9)

        #
        # L3.resize
        #
        blp, U, L0, L1, L2, L3, L4, L5 = tmp()
        L3.resize(nxR=2, nxZ=3, nxB=4)
        self.assertEqual(list(U.c[U]),  [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L0.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L1.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L2.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L3.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L4.c[U]), [-1, -1, -1, -1, -1, -1])
        self.assertEqual(list(L5.c[U]), [-1, -1, -1, -1, -1, -1])

        self.assertEqual([U.A[U].todok()[0,i] for i in range(U.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L0.A[U].todok()[0,i] for i in range(L0.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L1.A[U].todok()[0,i] for i in range(L1.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L2.A[U].todok()[0,i] for i in range(L2.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L3.A[U].todok()[0,i] for i in range(L3.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L4.A[U].todok()[0,i] for i in range(L4.A[U].shape[1])], [1,1,1,1,1,1])
        self.assertEqual([L5.A[U].todok()[0,i] for i in range(L5.A[U].shape[1])], [1,1,1,1,1,1])

        self.assertEqual(list(U.c[L0]),  [2,2,2,2,2,2,2])
        self.assertEqual(list(L0.c[L0]), [2,2,2,2,2,2,2])
        self.assertEqual(list(L2.c[L0]), [2,2,2,2,2,2,2])
        self.assertEqual(list(L3.c[L0]), [2,2,2,2,2,2,2])

        self.assertEqual([U.A[L0].todok()[0,i] for i in range(U.A[L0].shape[1])], [3]*7)
        self.assertEqual([L0.A[L0].todok()[0,i] for i in range(L0.A[L0].shape[1])], [3]*7)
        self.assertEqual([L2.A[L0].todok()[0,i] for i in range(L2.A[L0].shape[1])], [3]*7)
        self.assertEqual([L3.A[L0].todok()[0,i] for i in range(L3.A[L0].shape[1])], [3]*7)

        self.assertEqual(list(U.c[L3]),  [6,0,6,6,6,6,6,6,6])
        self.assertEqual(list(L0.c[L3]), [6,0,6,6,6,6,6,6,6])
        self.assertEqual(list(L3.c[L3]), [6,0,6,6,6,6,6,6,6])

        self.assertEqual([U.A[L3].todok()[0,i] for i in range(U.A[L3].shape[1])],   [7,0,7,7,7,7,7,7,7])
        self.assertEqual([L0.A[L3].todok()[0,i] for i in range(L0.A[L3].shape[1])], [7,0,7,7,7,7,7,7,7])
        self.assertEqual([L3.A[L3].todok()[0,i] for i in range(L3.A[L3].shape[1])], [7,0,7,7,7,7,7,7,7])

        self.assertEqual(list(U.c[L1]), [2]*8)
        self.assertEqual(list(U.c[L2]), [4]*8)
        self.assertEqual(list(U.c[L4]), [8]*8)
        self.assertEqual(list(U.c[L5]), [10]*9)

        self.assertEqual([U.A[L1].todok()[0,i] for i in range(U.A[L1].shape[1])], [3]*8)
        self.assertEqual([U.A[L2].todok()[0,i] for i in range(U.A[L2].shape[1])], [5]*8)
        self.assertEqual([U.A[L4].todok()[0,i] for i in range(U.A[L4].shape[1])], [9]*8)
        self.assertEqual([U.A[L5].todok()[0,i] for i in range(U.A[L5].shape[1])], [11]*9)

        self.assertEqual(list(L0.c[L2]), [4]*8)

        self.assertEqual([L0.A[L2].todok()[0,i] for i in range(L0.A[L2].shape[1])], [5]*8)

        self.assertEqual(list(L1.c[L1]), [2]*8)
        self.assertEqual(list(L1.c[L4]), [8]*8)
        self.assertEqual(list(L1.c[L5]), [10]*9)

        self.assertEqual([L1.A[L1].todok()[0,i] for i in range(L1.A[L1].shape[1])], [3]*8)
        self.assertEqual([L1.A[L4].todok()[0,i] for i in range(L1.A[L4].shape[1])], [9]*8)
        self.assertEqual([L1.A[L5].todok()[0,i] for i in range(L1.A[L5].shape[1])], [11]*9)

        self.assertEqual(list(L2.c[L2]), [4]*8)

        self.assertEqual([L2.A[L2].todok()[0,i] for i in range(L2.A[L2].shape[1])], [5]*8)

        self.assertEqual(list(L4.c[L1]), [2]*8)
        self.assertEqual(list(L4.c[L4]), [8]*8)

        self.assertEqual([L4.A[L1].todok()[0,i] for i in range(L4.A[L1].shape[1])], [3]*8)
        self.assertEqual([L4.A[L4].todok()[0,i] for i in range(L4.A[L4].shape[1])], [9]*8)

    def test_check_matrix_initialization(self):
        blp = LinearMultilevelProblem()
        U = blp.add_upper(nxR=2, nxZ=3, nxB=4)
        L = U.add_lower(nxR=1, nxZ=2, nxB=3)

        U.b = [1,2,3]

        U.A[U] = [[1,0,1,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0]]
        U.A[L] = [[1,1,0,1,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]]

        U.LL = [1,2,3,4]

        L.A[U] = [[1,0,1,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0]]
        L.A[L] = [[1,1,0,1,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]]

        self.assertEqual(U.A[U].shape, (3,9))

    def test_check_opposite_objectives(self):
        blp = LinearMultilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=1, nxB=1)
        L = U.add_lower(nxR=1, nxZ=1, nxB=1)
        U.c[U] = [1, 2, 3]
        U.c[L] = [4, 5, 6]
        U.d = 7
        L.c[U] = [-1, -2, -3]
        L.c[L] = [-4, -5, -6]
        U.d = -7
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c[U] = [1, 2, 3]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c[U] = [-1, -2, -3]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c[L] = [4, 5, 6]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c[L] = [-4, -5, -6]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c[L] = None
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        U.c[L] = None
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        blp = LinearMultilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=1, nxB=1)
        U.c[U] = [1, 2, 3]
        U.c[L] = [4, 5, 6]
        U.d = 7
        L = U.add_lower(nxR=1, nxZ=1, nxB=1)
        L.c = U.c
        L.minimize = False
        self.assertEqual( blp.check_opposite_objectives(U,L), True )


if __name__ == "__main__":
    unittest.main()
