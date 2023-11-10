import numpy as np
import scipy.sparse
import pyomo.common.unittest as unittest
from pao.mpr import *
from pao.mpr.repn import SimplifiedList, LinearLevelRepn, LevelVariable, LevelValues, LevelValueWrapper1, LevelValueWrapper2


class A(object):

    def __init__(self, i=0):
        self.i = i

    def clone(self, parent=None, clone_fn=None):
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


class Test_LevelValueWrapper1(unittest.TestCase):

    def test_init(self):
        l = LevelValueWrapper1('foo')
        self.assertEqual(l._matrix, False)
        self.assertEqual(l._values, {})
        self.assertEqual(l._prefix, 'foo')

    def test_init_matrix(self):
        l = LevelValueWrapper1('foo', matrix=True)
        self.assertEqual(l._matrix, True)
        self.assertEqual(l._values, {})
        self.assertEqual(l._prefix, 'foo')

    def test_len(self):
        l = LevelValueWrapper1('foo', matrix=True)
        self.assertEqual(len(l), 0)
        L0 = LinearLevelRepn(1,2,3)
        L1 = LinearLevelRepn(1,2,3)
        l[L0] = [1,2,3]
        l[L1] = [1,2,3]
        self.assertEqual(len(l), 2)

    def test_iter(self):
        l = LevelValueWrapper1('foo', matrix=True)
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
        l = LevelValueWrapper1('foo', matrix=True)
        try:
            l.foo
        except AttributeError:
            pass
        try:
            l.L
        except AttributeError:
            pass
        
    def test_setgetitem(self):
        l = LevelValueWrapper1('foo', matrix=False)
        L0 = LinearLevelRepn(1,2,3)
        L1 = LinearLevelRepn(1,2,3)
        l[L0] = [1,2,3]
        l[L1.id] = [4,5,6]
        self.assertEqual(len(l), 2)
        self.assertEqual(list(l[L0.id]), [1,2,3])
        self.assertEqual(list(l[L1]), [4,5,6])

    def test_clone(self):
        l = LevelValueWrapper1('foo', matrix=True)
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
        

class Test_LevelValueWrapper2(unittest.TestCase):

    def test_init_matrix(self):
        l = LevelValueWrapper2('foo')
        self.assertEqual(l._matrix, True)
        self.assertEqual(l._values, {})
        self.assertEqual(l._prefix, 'foo')

    def test_init_matrixlist(self):
        l = LevelValueWrapper2('foo', matrix=False)
        self.assertEqual(l._matrix, False)
        self.assertEqual(l._values, {})
        self.assertEqual(l._prefix, 'foo')

    def test_len_matrix(self):
        l = LevelValueWrapper2('foo', matrix=True)
        self.assertEqual(len(l), 0)
        L0 = LinearLevelRepn(1,2,3)
        L1 = L0.add_lower(nxR=1, nxZ=2, nxB=3, name="L1")
        L2 = L1.add_lower(nxR=1, nxZ=2, nxB=3, name="L2")
        l[L0,L1] = [1,2,3]
        l[L1,L2] = [1,2,3]
        self.assertEqual(len(l), 2)

    def test_iter(self):
        l = LevelValueWrapper2('foo', matrix=True)
        self.assertEqual(len(l), 0)
        L0 = LinearLevelRepn(1,2,3)
        L1 = L0.add_lower(nxR=1, nxZ=2, nxB=3, name="L1")
        L2 = L1.add_lower(nxR=1, nxZ=2, nxB=3, name="L2")
        L3 = L2.add_lower(nxR=1, nxZ=2, nxB=3, name="L3")
        l[L0,L1] = [1,2,3]
        l[L1,L2] = [1,2,3]
        l[L2,L3] = [1,2,3]
        self.assertEqual(len(l), 3)
        self.assertEqual(sorted(i for i in l), sorted([(L0.id,L1.id), (L1.id,L2.id), (L2.id,L3.id)]))

    def test_getattr(self):
        l = LevelValueWrapper2('foo', matrix=True)
        try:
            l.foo
        except AttributeError:
            pass
        try:
            l.L
        except AttributeError:
            pass
        
    def test_setgetitem_matrix(self):
        l = LevelValueWrapper2('foo', matrix=True)
        L0 = LinearLevelRepn(1,2,3)
        L1 = L0.add_lower(nxR=1, nxZ=2, nxB=3, name="L1")
        L2 = L1.add_lower(nxR=1, nxZ=2, nxB=3, name="L2")
        l[L0,L1] = [1,2,3]
        l[L1.id,L2.id] = [4,5,6]
        self.assertEqual(len(l), 2)
        self.assertTrue( np.array_equal(l[L0.id,L1.id].todense(), [[1,2,3]]) )
        self.assertTrue( np.array_equal(l[L1,L2].todense(), [[4,5,6]]) )

    def test_setgetitem_matrix_error1(self):
        try:
            l = LevelValueWrapper2('foo', matrix=True)
            L0 = LinearLevelRepn(1,2,3)
            L1 = LinearLevelRepn(1,2,3)
            l[L1,L0] = [1,2,3]
            self.fail("Expected assertion error")
        except AssertionError:
            pass

    def test_setgetitem_matrixlist(self):
        L = LevelValueWrapper2('foo', matrix=False)
        L0 = LinearLevelRepn(1,2,3)
        L1 = L0.add_lower(nxR=1, nxZ=2, nxB=3, name="L1")
        L2 = L1.add_lower(nxR=1, nxZ=2, nxB=3, name="L2")
        self.assertEqual(L[L0,L1], None)
        L[L0,L1] = (2,3,4), {(0,0,0):1, (1,1,1):2}
        L[L1.id,L2.id] = (2,3,4), {(0,1,2):1, (1,2,3):2}
        self.assertEqual(len(L), 2)
        self.assertTrue( np.array_equal(L[L0.id,L1.id][0].todense(), [[1,0,0,0],[0,0,0,0],[0,0,0,0]]) )
        self.assertTrue( np.array_equal(L[L0.id,L1.id][1].todense(), [[0,0,0,0],[0,2,0,0],[0,0,0,0]]) )
        self.assertTrue( np.array_equal(L[L1,L2][0].todense(), [[0,0,0,0],[0,0,1,0],[0,0,0,0]]) )
        self.assertTrue( np.array_equal(L[L1,L2][1].todense(), [[0,0,0,0],[0,0,0,0],[0,0,0,2]]) )
        L[L0,L1] = None
        self.assertEqual(L[L0,L1], None)

    def test_clone(self):
        l = LevelValueWrapper2('foo', matrix=True)
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

    def _create(self, *args, **kwargs):
        return LinearMultilevelProblem(*args, **kwargs)

    def test_init(self):
        blp = self._create()
        self.assertEqual(blp.name, None)
        
    def test_init_name(self):
        blp = self._create('foo')
        self.assertEqual(blp.name, 'foo')

    def test_names(self):
        blp = self._create()
        A = blp.add_upper(nxR=1, nxZ=2, nxB=3, name='A')
        B = A.add_lower(nxR=1, nxZ=2, nxB=3, name='B')
        C = A.add_lower(nxR=1, nxZ=2, nxB=3, name='C')
        self.assertEqual(A.name, 'A')
        self.assertEqual(B.name, 'B')
        self.assertEqual(C.name, 'C')

    def test_levels(self):
        blp = self._create()
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
        blp = self._create()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        self.assertEqual(len(U.x), 6)
        self.assertEqual(id(U), id(blp.U))

    def test_add_lower(self):
        blp = self._create()
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
        blp = self._create()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        L0 = U.add_lower(nxR=1, nxZ=2, nxB=4)
        L1 = U.add_lower(nxR=1, nxZ=2, nxB=5)

        U.c[U] = [0]*6

        U.A[U] = [[1]*6]
        U.b = [0]

        ans = blp.clone()

        self.assertEqual(len(ans.U.x), 6)
        self.assertEqual(len(U.LL[0].x), 7)
        self.assertEqual(len(U.LL[1].x), 8)

        self.assertEqual(len(ans.U.c[U]), 6)
        self.assertEqual(len(ans.U.c[U]), 6)

    def _create_tmp(self):
        blp = self._create()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        L0 = U.add_lower(nxR=1, nxZ=2, nxB=4, name='L0')
        L1 = U.add_lower(nxR=1, nxZ=2, nxB=5, name='L1')
        L2 = L0.add_lower(nxR=1, nxZ=3, nxB=4, name='L2')
        L3 = L0.add_lower(nxR=1, nxZ=4, nxB=4, name='L3')
        L4 = L1.add_lower(nxR=2, nxZ=2, nxB=4, name='L4')
        L5 = L1.add_lower(nxR=3, nxZ=2, nxB=4, name='L5')

        # U
        U.c[U] = [-1]*6
        U.c[L0] = [2]*7
        U.c[L1] = [2]*8
        U.c[L2] = [4]*8
        U.c[L3] = [6]*9
        U.c[L4] = [8]*8
        U.c[L5] = [10]*9

        U.A[U] = [[1]*6]
        U.A[L0] = [[3]*7]
        U.A[L1] = [[3]*8]
        U.A[L2] = [[5]*8]
        U.A[L3] = [[7]*9]
        U.A[L4] = [[9]*8]
        U.A[L5] = [[11]*9]
        U.b = [0]

        # L0
        L0.c[U] = [-1]*6
        L0.c[L0] = [2]*7
        L0.c[L2] = [4]*8
        L0.c[L3] = [6]*9

        L0.A[U] = [[1]*6]
        L0.A[L0] = [[3]*7]
        L0.A[L2] = [[5]*8]
        L0.A[L3] = [[7]*9]
        U.b = [1]

        # L2
        L2.c[U] = [-1]*6
        L2.c[L0] = [2]*7
        L2.c[L2] = [4]*8

        L2.A[U] = [[1]*6]
        L2.A[L0] = [[3]*7]
        L2.A[L2] = [[5]*8]
        U.b = [2]

        # L3
        L3.c[U] = [-1]*6
        L3.c[L0] = [2]*7
        L3.c[L3] = [6]*9

        L3.A[U] = [[1]*6]
        L3.A[L0] = [[3]*7]
        L3.A[L3] = [[7]*9]
        U.b = [3]

        # L1
        L1.c[U] = [-1]*6
        L1.c[L1] = [2]*8
        L1.c[L4] = [8]*8
        L1.c[L5] = [10]*9

        L1.A[U] = [[1]*6]
        L1.A[L1] = [[3]*8]
        L1.A[L4] = [[9]*8]
        L1.A[L5] = [[11]*9]
        U.b = [4]

        # L4
        L4.c[U] = [-1]*6
        L4.c[L1] = [2]*8
        L4.c[L4] = [8]*8

        L4.A[U] = [[1]*6]
        L4.A[L1] = [[3]*8]
        L4.A[L4] = [[9]*8]
        U.b = [5]

        # L5
        L5.c[U] = [-1]*6
        L5.c[L1] = [2]*8
        L5.c[L5] = [10]*9

        L5.A[U] = [[1]*6]
        L5.A[L1] = [[3]*8]
        L5.A[L5] = [[11]*9]
        U.b = [6]

        return blp, U, L0, L1, L2, L3, L4, L5

    def test_resize_U(self):
        #
        # U.resize
        #
        blp, U, L0, L1, L2, L3, L4, L5 = self._create_tmp()
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

    def test_resize_L0(self):
        #
        # L0.resize
        #
        blp, U, L0, L1, L2, L3, L4, L5 = self._create_tmp()
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

    def test_resize_L3(self):
        #
        # L3.resize
        #
        blp, U, L0, L1, L2, L3, L4, L5 = self._create_tmp()
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
        blp = self._create()
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
        blp = self._create()
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

        blp = self._create()
        U = blp.add_upper(nxR=1, nxZ=1, nxB=1)
        U.c[U] = [1, 2, 3]
        U.c[L] = [4, 5, 6]
        U.d = 7
        L = U.add_lower(nxR=1, nxZ=1, nxB=1)
        L.c = U.c
        L.maximize = True
        self.assertEqual( blp.check_opposite_objectives(U,L), True )


class Test_QuadraticMultilevelProblem(Test_LinearMultilevelProblem):

    def _create(self, *args, **kwargs):
        return QuadraticMultilevelProblem(*args, **kwargs)

    def test_clone(self):
        blp = self._create()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        L0 = U.add_lower(nxR=1, nxZ=2, nxB=4)
        L1 = U.add_lower(nxR=1, nxZ=2, nxB=5)

        U.c[U] = [0]*6
        U.P[U,L0] = (6,7), {(i,i):i+1 for i in range(6)}
        U.P[U,L1] = (6,8), {(i,i):i+2 for i in range(6)}

        U.A[U] = [[1]*6, [2]*6]
        U.Q[U,L0] = (2,6,7), {(0,i,i):i+3 for i in range(6)}
        U.Q[U,L1] = (2,6,8), {(0,i,i):i+4 for i in range(6)}
        U.b = [0,0]

        ans = blp.clone()

        self.assertEqual(len(ans.U.x), 6)
        self.assertEqual(len(U.LL[0].x), 7)
        self.assertEqual(len(U.LL[1].x), 8)

        self.assertEqual(len(ans.U.c[U]), 6)
        self.assertEqual(len(ans.U.c[U]), 6)

    def _create_tmp(self):
        blp, U, L0, L1, L2, L3, L4, L5 = Test_LinearMultilevelProblem._create_tmp(self)
        # U - L0 - L1
        # U - L0 - L2
        # U - L3 - L4
        # U - L3 - L5

        # U  = 6 
        # L0 = 7
        # L1 = 8
        # L2 = 8
        # L3 = 9
        # L4 = 8
        # L5 = 9

        # U
        U.P[U,L0] =   (6,7), {(i,i):1 for i in range(6)}
        U.P[U,L1] =   (6,8), {(i,i):2 for i in range(6)}
        U.P[U,L2] =   (6,8), {(i,i):3 for i in range(6)}
        U.P[U,L3] =   (6,9), {(i,i):4 for i in range(6)}
        U.P[U,L4] =   (6,8), {(i,i):5 for i in range(6)}
        U.P[U,L5] =   (6,9), {(i,i):6 for i in range(6)}
        U.P[L0,L2] =  (7,8), {(i,i):7 for i in range(7)}
        U.P[L0,L3] =  (7,8), {(i,i):8 for i in range(7)}
        U.P[L1,L4] =  (9,8), {(i,i):9 for i in range(8)}
        U.P[L1,L5] =  (9,9), {(i,i):10 for i in range(9)}

        # L0
        L0.P[U,L0] =   (6,7), {(i,i):1 for i in range(6)}
        L0.P[U,L2] =   (6,8), {(i,i):3 for i in range(6)}
        L0.P[U,L3] =   (6,9), {(i,i):4 for i in range(6)}
        L0.P[L0,L2] =  (7,8), {(i,i):7 for i in range(7)}
        L0.P[L0,L3] =  (7,8), {(i,i):8 for i in range(7)}

        # L2
        L2.P[U,L0] =   (6,7), {(i,i):1 for i in range(6)}
        L2.P[U,L2] =   (6,8), {(i,i):3 for i in range(6)}
        L2.P[L0,L2] =  (7,8), {(i,i):7 for i in range(7)}

        # L1 
        L1.P[U,L1] =   (6,8), {(i,i):2 for i in range(6)}
        L1.P[U,L4] =   (6,8), {(i,i):5 for i in range(6)}
        L1.P[U,L5] =   (6,9), {(i,i):6 for i in range(6)}
        L1.P[L1,L4] =  (9,8), {(i,i):9 for i in range(8)}
        L1.P[L1,L5] =  (9,9), {(i,i):10 for i in range(9)}

        # L5 
        L5.P[U,L1] =   (6,8), {(i,i):2 for i in range(6)}
        L5.P[U,L5] =   (6,9), {(i,i):6 for i in range(6)}
        L5.P[L1,L5] =  (9,9), {(i,i):10 for i in range(9)}


        # U
        U.Q[U,L0] =   (1,6,7), {(0,i,i):1 for i in range(6)}
        U.Q[U,L1] =   (1,6,8), {(0,i,i):2 for i in range(6)}
        U.Q[U,L2] =   (1,6,8), {(0,i,i):3 for i in range(6)}
        U.Q[U,L3] =   (1,6,9), {(0,i,i):4 for i in range(6)}
        U.Q[U,L4] =   (1,6,8), {(0,i,i):5 for i in range(6)}
        U.Q[U,L5] =   (1,6,9), {(0,i,i):6 for i in range(6)}
        U.Q[L0,L2] =  (1,7,8), {(0,i,i):7 for i in range(7)}
        U.Q[L0,L3] =  (1,7,8), {(0,i,i):8 for i in range(7)}
        U.Q[L1,L4] =  (1,9,8), {(0,i,i):9 for i in range(8)}
        U.Q[L1,L5] =  (1,9,9), {(0,i,i):10 for i in range(9)}

        # L0
        L0.Q[U,L0] =   (1,6,7), {(0,i,i):1 for i in range(6)}
        L0.Q[U,L2] =   (1,6,8), {(0,i,i):3 for i in range(6)}
        L0.Q[U,L3] =   (1,6,9), {(0,i,i):4 for i in range(6)}
        L0.Q[L0,L2] =  (1,7,8), {(0,i,i):7 for i in range(7)}
        L0.Q[L0,L3] =  (1,7,8), {(0,i,i):8 for i in range(7)}

        # L2
        L2.Q[U,L0] =   (1,6,7), {(0,i,i):1 for i in range(6)}
        L2.Q[U,L2] =   (1,6,8), {(0,i,i):3 for i in range(6)}
        L2.Q[L0,L2] =  (1,7,8), {(0,i,i):7 for i in range(7)}

        # L1 
        L1.Q[U,L1] =   (1,6,8), {(0,i,i):2 for i in range(6)}
        L1.Q[U,L4] =   (1,6,8), {(0,i,i):5 for i in range(6)}
        L1.Q[U,L5] =   (1,6,9), {(0,i,i):6 for i in range(6)}
        L1.Q[L1,L4] =  (1,9,8), {(0,i,i):9 for i in range(8)}
        L1.Q[L1,L5] =  (1,9,9), {(0,i,i):10 for i in range(9)}

        # L5 
        L5.Q[U,L1] =   (1,6,8), {(0,i,i):2 for i in range(6)}
        L5.Q[U,L5] =   (1,6,9), {(0,i,i):6 for i in range(6)}
        L5.Q[L1,L5] =  (1,9,9), {(0,i,i):10 for i in range(9)}


        return blp, U, L0, L1, L2, L3, L4, L5

    def test_resize_U(self):
        blp, U, L0, L1, L2, L3, L4, L5 = self._create_tmp()
        U.resize(nxR=2, nxZ=3, nxB=4)

        self.assertEqual(dict(U.P[U,L0].todok()),  {(0, 0):1, (2, 1):1, (3, 2):1, (5, 3):1, (6, 4):1, (7, 5):1})
        self.assertEqual(dict(U.P[U,L1].todok()),  {(0, 0):2, (2, 1):2, (3, 2):2, (5, 3):2, (6, 4):2, (7, 5):2})
        self.assertEqual(dict(U.P[U,L2].todok()),  {(0, 0):3, (2, 1):3, (3, 2):3, (5, 3):3, (6, 4):3, (7, 5):3})
        self.assertEqual(dict(U.P[U,L3].todok()),  {(0, 0):4, (2, 1):4, (3, 2):4, (5, 3):4, (6, 4):4, (7, 5):4})
        self.assertEqual(dict(U.P[U,L4].todok()),  {(0, 0):5, (2, 1):5, (3, 2):5, (5, 3):5, (6, 4):5, (7, 5):5})
        self.assertEqual(dict(U.P[U,L5].todok()),  {(0, 0):6, (2, 1):6, (3, 2):6, (5, 3):6, (6, 4):6, (7, 5):6})
        self.assertEqual(dict(U.P[L0,L2].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(U.P[L0,L3].todok()), {(0, 0):8, (1, 1):8, (2, 2):8, (3, 3):8, (4, 4):8, (5, 5):8, (6, 6):8})
        self.assertEqual(dict(U.P[L1,L4].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(U.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(U.Q[U,L0][0].todok()),  {(0, 0):1, (2, 1):1, (3, 2):1, (5, 3):1, (6, 4):1, (7, 5):1})
        self.assertEqual(dict(U.Q[U,L1][0].todok()),  {(0, 0):2, (2, 1):2, (3, 2):2, (5, 3):2, (6, 4):2, (7, 5):2})
        self.assertEqual(dict(U.Q[U,L2][0].todok()),  {(0, 0):3, (2, 1):3, (3, 2):3, (5, 3):3, (6, 4):3, (7, 5):3})
        self.assertEqual(dict(U.Q[U,L3][0].todok()),  {(0, 0):4, (2, 1):4, (3, 2):4, (5, 3):4, (6, 4):4, (7, 5):4})
        self.assertEqual(dict(U.Q[U,L4][0].todok()),  {(0, 0):5, (2, 1):5, (3, 2):5, (5, 3):5, (6, 4):5, (7, 5):5})
        self.assertEqual(dict(U.Q[U,L5][0].todok()),  {(0, 0):6, (2, 1):6, (3, 2):6, (5, 3):6, (6, 4):6, (7, 5):6})
        self.assertEqual(dict(U.Q[L0,L2][0].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(U.Q[L0,L3][0].todok()), {(0, 0):8, (1, 1):8, (2, 2):8, (3, 3):8, (4, 4):8, (5, 5):8, (6, 6):8})
        self.assertEqual(dict(U.Q[L1,L4][0].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(U.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


        self.assertEqual(dict(L0.P[U,L0].todok()),  {(0, 0):1, (2, 1):1, (3, 2):1, (5, 3):1, (6, 4):1, (7, 5):1})
        self.assertEqual(dict(L0.P[U,L2].todok()),  {(0, 0):3, (2, 1):3, (3, 2):3, (5, 3):3, (6, 4):3, (7, 5):3})
        self.assertEqual(dict(L0.P[U,L3].todok()),  {(0, 0):4, (2, 1):4, (3, 2):4, (5, 3):4, (6, 4):4, (7, 5):4})
        self.assertEqual(dict(L0.P[L0,L2].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(L0.P[L0,L3].todok()), {(0, 0):8, (1, 1):8, (2, 2):8, (3, 3):8, (4, 4):8, (5, 5):8, (6, 6):8})

        self.assertEqual(dict(L0.Q[U,L0][0].todok()),  {(0, 0):1, (2, 1):1, (3, 2):1, (5, 3):1, (6, 4):1, (7, 5):1})
        self.assertEqual(dict(L0.Q[U,L2][0].todok()),  {(0, 0):3, (2, 1):3, (3, 2):3, (5, 3):3, (6, 4):3, (7, 5):3})
        self.assertEqual(dict(L0.Q[U,L3][0].todok()),  {(0, 0):4, (2, 1):4, (3, 2):4, (5, 3):4, (6, 4):4, (7, 5):4})
        self.assertEqual(dict(L0.Q[L0,L2][0].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(L0.Q[L0,L3][0].todok()), {(0, 0):8, (1, 1):8, (2, 2):8, (3, 3):8, (4, 4):8, (5, 5):8, (6, 6):8})


        self.assertEqual(dict(L2.P[U,L0].todok()),  {(0, 0):1, (2, 1):1, (3, 2):1, (5, 3):1, (6, 4):1, (7, 5):1})
        self.assertEqual(dict(L2.P[U,L2].todok()),  {(0, 0):3, (2, 1):3, (3, 2):3, (5, 3):3, (6, 4):3, (7, 5):3})
        self.assertEqual(dict(L2.P[L0,L2].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})

        self.assertEqual(dict(L2.Q[U,L0][0].todok()),  {(0, 0):1, (2, 1):1, (3, 2):1, (5, 3):1, (6, 4):1, (7, 5):1})
        self.assertEqual(dict(L2.Q[U,L2][0].todok()),  {(0, 0):3, (2, 1):3, (3, 2):3, (5, 3):3, (6, 4):3, (7, 5):3})
        self.assertEqual(dict(L2.Q[L0,L2][0].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})


        self.assertEqual(dict(L1.P[U,L1].todok()),  {(0, 0):2, (2, 1):2, (3, 2):2, (5, 3):2, (6, 4):2, (7, 5):2})
        self.assertEqual(dict(L1.P[U,L4].todok()),  {(0, 0):5, (2, 1):5, (3, 2):5, (5, 3):5, (6, 4):5, (7, 5):5})
        self.assertEqual(dict(L1.P[U,L5].todok()),  {(0, 0):6, (2, 1):6, (3, 2):6, (5, 3):6, (6, 4):6, (7, 5):6})
        self.assertEqual(dict(L1.P[L1,L4].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(L1.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(L1.Q[U,L1][0].todok()),  {(0, 0):2, (2, 1):2, (3, 2):2, (5, 3):2, (6, 4):2, (7, 5):2})
        self.assertEqual(dict(L1.Q[U,L4][0].todok()),  {(0, 0):5, (2, 1):5, (3, 2):5, (5, 3):5, (6, 4):5, (7, 5):5})
        self.assertEqual(dict(L1.Q[U,L5][0].todok()),  {(0, 0):6, (2, 1):6, (3, 2):6, (5, 3):6, (6, 4):6, (7, 5):6})
        self.assertEqual(dict(L1.Q[L1,L4][0].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(L1.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


        self.assertEqual(dict(L5.P[U,L1].todok()),  {(0, 0):2, (2, 1):2, (3, 2):2, (5, 3):2, (6, 4):2, (7, 5):2})
        self.assertEqual(dict(L5.P[U,L5].todok()),  {(0, 0):6, (2, 1):6, (3, 2):6, (5, 3):6, (6, 4):6, (7, 5):6})
        self.assertEqual(dict(L5.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(L5.Q[U,L1][0].todok()),  {(0, 0):2, (2, 1):2, (3, 2):2, (5, 3):2, (6, 4):2, (7, 5):2})
        self.assertEqual(dict(L5.Q[U,L5][0].todok()),  {(0, 0):6, (2, 1):6, (3, 2):6, (5, 3):6, (6, 4):6, (7, 5):6})
        self.assertEqual(dict(L5.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

    def test_resize_L0(self):
        blp, U, L0, L1, L2, L3, L4, L5 = self._create_tmp()
        L0.resize(nxR=2, nxZ=3, nxB=4)

        self.assertEqual(dict(U.P[U,L0].todok()),  {(0, 0):1, (1, 2):1, (2, 3):1, (3, 5):1, (4, 6):1, (5, 7):1})
        self.assertEqual(dict(U.P[U,L1].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(U.P[U,L2].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(U.P[U,L3].todok()),  {(0, 0):4, (1, 1):4, (2, 2):4, (3, 3):4, (4, 4):4, (5, 5):4})
        self.assertEqual(dict(U.P[U,L4].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(U.P[U,L5].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(U.P[L0,L2].todok()), {(0, 0):7, (2, 1):7, (3, 2):7, (5, 3):7, (6, 4):7, (7, 5):7, (8, 6):7})
        self.assertEqual(dict(U.P[L0,L3].todok()), {(0, 0):8, (2, 1):8, (3, 2):8, (5, 3):8, (6, 4):8, (7, 5):8, (8, 6):8})
        self.assertEqual(dict(U.P[L1,L4].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(U.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(U.Q[U,L0][0].todok()),  {(0, 0):1, (1, 2):1, (2, 3):1, (3, 5):1, (4, 6):1, (5, 7):1})
        self.assertEqual(dict(U.Q[U,L1][0].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(U.Q[U,L2][0].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(U.Q[U,L3][0].todok()),  {(0, 0):4, (1, 1):4, (2, 2):4, (3, 3):4, (4, 4):4, (5, 5):4})
        self.assertEqual(dict(U.Q[U,L4][0].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(U.Q[U,L5][0].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(U.Q[L0,L2][0].todok()), {(0, 0):7, (2, 1):7, (3, 2):7, (5, 3):7, (6, 4):7, (7, 5):7, (8, 6):7})
        self.assertEqual(dict(U.Q[L0,L3][0].todok()), {(0, 0):8, (2, 1):8, (3, 2):8, (5, 3):8, (6, 4):8, (7, 5):8, (8, 6):8})
        self.assertEqual(dict(U.Q[L1,L4][0].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(U.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


        self.assertEqual(dict(L0.P[U,L0].todok()),  {(0, 0):1, (1, 2):1, (2, 3):1, (3, 5):1, (4, 6):1, (5, 7):1})
        self.assertEqual(dict(L0.P[U,L2].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L0.P[U,L3].todok()),  {(0, 0):4, (1, 1):4, (2, 2):4, (3, 3):4, (4, 4):4, (5, 5):4})
        self.assertEqual(dict(L0.P[L0,L2].todok()), {(0, 0):7, (2, 1):7, (3, 2):7, (5, 3):7, (6, 4):7, (7, 5):7, (8, 6):7})
        self.assertEqual(dict(L0.P[L0,L3].todok()), {(0, 0):8, (2, 1):8, (3, 2):8, (5, 3):8, (6, 4):8, (7, 5):8, (8, 6):8})

        self.assertEqual(dict(L0.Q[U,L0][0].todok()),  {(0, 0):1, (1, 2):1, (2, 3):1, (3, 5):1, (4, 6):1, (5, 7):1})
        self.assertEqual(dict(L0.Q[U,L2][0].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L0.Q[U,L3][0].todok()),  {(0, 0):4, (1, 1):4, (2, 2):4, (3, 3):4, (4, 4):4, (5, 5):4})
        self.assertEqual(dict(L0.Q[L0,L2][0].todok()), {(0, 0):7, (2, 1):7, (3, 2):7, (5, 3):7, (6, 4):7, (7, 5):7, (8, 6):7})
        self.assertEqual(dict(L0.Q[L0,L3][0].todok()), {(0, 0):8, (2, 1):8, (3, 2):8, (5, 3):8, (6, 4):8, (7, 5):8, (8, 6):8})


        self.assertEqual(dict(L2.P[U,L0].todok()),  {(0, 0):1, (1, 2):1, (2, 3):1, (3, 5):1, (4, 6):1, (5, 7):1})
        self.assertEqual(dict(L2.P[U,L2].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L2.P[L0,L2].todok()), {(0, 0):7, (2, 1):7, (3, 2):7, (5, 3):7, (6, 4):7, (7, 5):7, (8, 6):7})

        self.assertEqual(dict(L2.Q[U,L0][0].todok()),  {(0, 0):1, (1, 2):1, (2, 3):1, (3, 5):1, (4, 6):1, (5, 7):1})
        self.assertEqual(dict(L2.Q[U,L2][0].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L2.Q[L0,L2][0].todok()), {(0, 0):7, (2, 1):7, (3, 2):7, (5, 3):7, (6, 4):7, (7, 5):7, (8, 6):7})


        self.assertEqual(dict(L1.P[U,L1].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L1.P[U,L4].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(L1.P[U,L5].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L1.P[L1,L4].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(L1.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(L1.Q[U,L1][0].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L1.Q[U,L4][0].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(L1.Q[U,L5][0].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L1.Q[L1,L4][0].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(L1.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


        self.assertEqual(dict(L5.P[U,L1].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L5.P[U,L5].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L5.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(L5.Q[U,L1][0].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L5.Q[U,L5][0].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L5.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


    def test_resize_L3(self):
        blp, U, L0, L1, L2, L3, L4, L5 = self._create_tmp()
        L3.resize(nxR=2, nxZ=3, nxB=4)

        self.assertEqual(dict(U.P[U,L0].todok()),  {(0, 0):1, (1, 1):1, (2, 2):1, (3, 3):1, (4, 4):1, (5, 5):1})
        self.assertEqual(dict(U.P[U,L1].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(U.P[U,L2].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(U.P[U,L3].todok()),  {(0, 0):4, (1, 2):4, (2, 3):4, (3, 4):4, (5, 5):4})          
        self.assertEqual(dict(U.P[U,L4].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(U.P[U,L5].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(U.P[L0,L2].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(U.P[L0,L3].todok()), {(0, 0):8, (1, 2):8, (2, 3):8, (3, 4):8, (5, 5):8, (6, 6):8})
        self.assertEqual(dict(U.P[L1,L4].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(U.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(U.Q[U,L0][0].todok()),  {(0, 0):1, (1, 1):1, (2, 2):1, (3, 3):1, (4, 4):1, (5, 5):1})
        self.assertEqual(dict(U.Q[U,L1][0].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(U.Q[U,L2][0].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(U.Q[U,L3][0].todok()),  {(0, 0):4, (1, 2):4, (2, 3):4, (3, 4):4, (5, 5):4})          
        self.assertEqual(dict(U.Q[U,L4][0].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(U.Q[U,L5][0].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(U.Q[L0,L2][0].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(U.Q[L0,L3][0].todok()), {(0, 0):8, (1, 2):8, (2, 3):8, (3, 4):8, (5, 5):8, (6, 6):8})
        self.assertEqual(dict(U.Q[L1,L4][0].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(U.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


        self.assertEqual(dict(L0.P[U,L0].todok()),  {(0, 0):1, (1, 1):1, (2, 2):1, (3, 3):1, (4, 4):1, (5, 5):1})
        self.assertEqual(dict(L0.P[U,L2].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L0.P[U,L3].todok()),  {(0, 0):4, (1, 2):4, (2, 3):4, (3, 4):4, (5, 5):4})           
        self.assertEqual(dict(L0.P[L0,L2].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(L0.P[L0,L3].todok()), {(0, 0):8, (1, 2):8, (2, 3):8, (3, 4):8, (5, 5):8, (6, 6):8})

        self.assertEqual(dict(L0.Q[U,L0][0].todok()),  {(0, 0):1, (1, 1):1, (2, 2):1, (3, 3):1, (4, 4):1, (5, 5):1})
        self.assertEqual(dict(L0.Q[U,L2][0].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L0.Q[U,L3][0].todok()),  {(0, 0):4, (1, 2):4, (2, 3):4, (3, 4):4, (5, 5):4})         
        self.assertEqual(dict(L0.Q[L0,L2][0].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})
        self.assertEqual(dict(L0.Q[L0,L3][0].todok()), {(0, 0):8, (1, 2):8, (2, 3):8, (3, 4):8, (5, 5):8, (6, 6):8}) 


        self.assertEqual(dict(L2.P[U,L0].todok()),  {(0, 0):1, (1, 1):1, (2, 2):1, (3, 3):1, (4, 4):1, (5, 5):1})
        self.assertEqual(dict(L2.P[U,L2].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L2.P[L0,L2].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})

        self.assertEqual(dict(L2.Q[U,L0][0].todok()),  {(0, 0):1, (1, 1):1, (2, 2):1, (3, 3):1, (4, 4):1, (5, 5):1})
        self.assertEqual(dict(L2.Q[U,L2][0].todok()),  {(0, 0):3, (1, 1):3, (2, 2):3, (3, 3):3, (4, 4):3, (5, 5):3})
        self.assertEqual(dict(L2.Q[L0,L2][0].todok()), {(0, 0):7, (1, 1):7, (2, 2):7, (3, 3):7, (4, 4):7, (5, 5):7, (6, 6):7})


        self.assertEqual(dict(L1.P[U,L1].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L1.P[U,L4].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(L1.P[U,L5].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L1.P[L1,L4].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(L1.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(L1.Q[U,L1][0].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L1.Q[U,L4][0].todok()),  {(0, 0):5, (1, 1):5, (2, 2):5, (3, 3):5, (4, 4):5, (5, 5):5})
        self.assertEqual(dict(L1.Q[U,L5][0].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L1.Q[L1,L4][0].todok()), {(0, 0):9, (1, 1):9, (2, 2):9, (3, 3):9, (4, 4):9, (5, 5):9, (6, 6):9, (7, 7):9})
        self.assertEqual(dict(L1.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


        self.assertEqual(dict(L5.P[U,L1].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L5.P[U,L5].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L5.P[L1,L5].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})

        self.assertEqual(dict(L5.Q[U,L1][0].todok()),  {(0, 0):2, (1, 1):2, (2, 2):2, (3, 3):2, (4, 4):2, (5, 5):2})
        self.assertEqual(dict(L5.Q[U,L5][0].todok()),  {(0, 0):6, (1, 1):6, (2, 2):6, (3, 3):6, (4, 4):6, (5, 5):6})
        self.assertEqual(dict(L5.Q[L1,L5][0].todok()), {(0, 0):10, (1, 1):10, (2, 2):10, (3, 3):10, (4, 4):10, (5, 5):10, (6, 6):10, (7, 7):10, (8, 8):10})


if __name__ == "__main__":
    unittest.main()
