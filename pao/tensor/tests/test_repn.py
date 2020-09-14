import numpy as np
import scipy.sparse
import pyutilib.th as unittest
from pao.tensor import *


class Test_SimplifiedList(unittest.TestCase):

    def test_init(self):
        l = SimplifiedList()
        self.assertEqual(len(l._data),0)

    def test_insert(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList()
        l.append(A())
        self.assertEqual(len(l), 1)
        l.insert(2, A(2))
        self.assertEqual(len(l), 2)
        
    def test_iter(self):
        l = SimplifiedList()
        l.append(0)
        l.append(2)
        l.append(1)
        self.assertEqual([v for v in l], [0,2,1])

    def test_getitem(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList()
        l.append(A())
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].i, 0)
        self.assertEqual(len(l), 1)
        try:
            l[2].i
            self.fail("Expected IndexError")
        except IndexError:
            pass
        
    def test_getitem_clone(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList(clone=A(1))
        l.append(A())
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].i, 0)
        self.assertEqual(len(l), 1)

        self.assertEqual(l[2].i, 1)
        self.assertEqual(len(l), 3)
        self.assertEqual(l[1].i, 1)
        self.assertEqual(len(l), 3)
        
    def test_setitem(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList()
        l.append(A())
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].i, 0)
        self.assertEqual(len(l), 1)

        try:
            l[3] = A(3)
            self.fail("Expected IndexError")
        except IndexError:
            pass

    def test_delitem(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
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
        l.append(1)
        l.append(2)
        self.assertEqual(len(l), len(l._data))

    def test_getattr(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList()
        try:
            l.i
            self.fail("Expected AssertionError")
        except:
            pass
       
        l.append(A(1))
        self.assertEqual(l.i, 1) 

        l.append(A(2))
        try:
            l.i
            self.fail("Expected AssertionError")
        except:
            pass

        self.assertEqual( getattr(l, '_foo', None), None )

    def test_setattr(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList()
        try:
            l.i = 0
            self.fail("Expected AssertionError")
        except:
            pass

        l.append(A(1))
        l.i = 0

        l.append(A(1))
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

    def test_setattr(self):
        l = LevelVariable(10)
        l.a = [1,2,3]
        self.assertEqual(type(l.a),list)
        l.lower_bounds = [1,2,3]
        self.assertEqual(type(l.lower_bounds),np.ndarray)


class Test_LevelValues(unittest.TestCase):

    def test_init(self):
        l = LevelValues()
        self.assertEqual(l._matrix, False)
        self.assertEqual(l.xR, None)
        self.assertEqual(l.xZ, None)
        self.assertEqual(l.xB, None)

    def test_init_matrix(self):
        l = LevelValues(matrix=True)
        self.assertEqual(l._matrix, True)
        self.assertEqual(l.xR, None)
        self.assertEqual(l.xZ, None)
        self.assertEqual(l.xB, None)

    def test_set_values(self):
        l = LevelValues(matrix=True)
        l.set_values(xR=1, xZ=2, xB=3)
        self.assertEqual(l.xR, 1)
        self.assertEqual(l.xZ, 2)
        self.assertEqual(l.xB, 3)

    def test_len(self):
        l = LevelValues()
        l.set_values(xR=[0,1,2], xZ=[3,4], xB=[5])
        self.assertEqual(len(l), 6)
        
    def test_len_matrix(self):
        l = LevelValues(matrix=True)
        l.set_values(xR=[(0,1,2),(3,4,5)], xB=[(0,0,5),(1,0,6)], xZ=[(1,1,1)])
        self.assertEqual(len(l), 4)
        l.set_values(xB=[(0,1,2),(3,4,5)], xR=[(0,0,5),(1,0,6)], xZ=[(1,1,1)])
        self.assertEqual(len(l), 4)
        
    def test_setattr(self):
        l = LevelValues()
        l.a = [1,2,3]
        self.assertEqual(type(l.a),list)
        l.xR = [1,2,3]
        self.assertEqual(type(l.xR),np.ndarray)

    def test_setattr_matrix(self):
        l = LevelValues(matrix=True)
        l.a = [(0,1,2),(3,4,5)]
        self.assertEqual(type(l.a),list)
        l.xR = [(0,1,2),(3,4,5)]
        self.assertEqual(type(l.xR),scipy.sparse.coo.coo_matrix)


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
        l.L
        self.assertEqual(len(l), 1)

    def test_getattr(self):
        l = LevelValueWrapper('foo', matrix=True)
        l.foo
        self.assertEqual(list(l._values.keys()), ['foo'])
        self.assertEqual(type(l.foo), LevelValues)
        l.L
        self.assertEqual(len(l.L),1)
        self.assertEqual(set(l._values.keys()), set(['foo','L']))
        #
        getattr(l, "_foo", False)
        

class Test_LinearLevelRepn(unittest.TestCase):

    def test_init(self):
        l = LinearLevelRepn(1,2,3)
        self.assertEqual(len(l.xR), 1)
        self.assertEqual(len(l.xZ), 2)
        self.assertEqual(len(l.xB), 3)
        self.assertEqual(l.minimize, True)
        self.assertEqual(len(l.c), 0)
        self.assertEqual(l.d, 0)
        self.assertEqual(len(l.A), 0)
        self.assertEqual(l.b.size, 0)
        self.assertEqual(l.inequalities, True)

    def test_setattr(self):
        l = LinearLevelRepn(1,2,3)
        l.x = -1
        self.assertEqual(l.x, -1)
        l.b = -1
        self.assertEqual(l.b, np.array(-1))
        

class Test_LinearBilevelProblem(unittest.TestCase):

    def test_init(self):
        blp = LinearBilevelProblem()
        self.assertEqual(blp.name, None)
        self.assertEqual(type(blp.L), SimplifiedList)
        
    def test_init_name(self):
        blp = LinearBilevelProblem('foo')
        self.assertEqual(blp.name, 'foo')

    def test_add_upper(self):
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        self.assertEqual(len(U.xR), 1)
        self.assertEqual(len(U.xZ), 2)
        self.assertEqual(len(U.xB), 3)
        self.assertEqual(id(U), id(blp.U))

    def test_add_lower(self):
        blp = LinearBilevelProblem()
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        self.assertEqual(len(L.xR), 1)
        self.assertEqual(len(L.xZ), 2)
        self.assertEqual(len(L.xB), 3)
        self.assertEqual(id(L), id(blp.L))

        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        self.assertEqual(len(L[1].xR), 1)
        self.assertEqual(len(L[1].xZ), 2)
        self.assertEqual(len(L[1].xB), 3)
        self.assertEqual(id(L[0]), id(blp.L[0]))
        self.assertEqual(id(L[1]), id(blp.L[1]))

    def test_check_opposite_objectives(self):
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=1, nxB=1)
        U.c.U.xR = [1]
        U.c.U.xZ = [2]
        U.c.U.xB = [3]
        U.c.L.xR = [4]
        U.c.L.xZ = [5]
        U.c.L.xB = [6]
        U.d = 7
        L = blp.add_lower(nxR=1, nxZ=1, nxB=1)
        L.c.U.xR = [-1]
        L.c.U.xZ = [-2]
        L.c.U.xB = [-3]
        L.c.L.xR = [-4]
        L.c.L.xZ = [-5]
        L.c.L.xB = [-6]
        U.d = -7
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.U.xR = [1]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c.U.xR = [-1]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.U.xZ = [2]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c.U.xZ = [-2]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.U.xB = [3]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c.U.xB = [-3]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.L.xR = [4]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c.L.xR = [-4]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.L.xZ = [5]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c.L.xZ = [-5]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.L.xB = [6]
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        L.c.L.xB = [-6]
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        L.c.L.xB = None
        self.assertEqual( blp.check_opposite_objectives(U,L), False )
        U.c.L.xB = None
        self.assertEqual( blp.check_opposite_objectives(U,L), True )

        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=1, nxB=1)
        U.c.U.xR = [1]
        U.c.U.xZ = [2]
        U.c.U.xB = [3]
        U.c.L.xR = [4]
        U.c.L.xZ = [5]
        U.c.L.xB = [6]
        U.d = 7
        L = blp.add_lower(nxR=1, nxZ=1, nxB=1)
        L.c = U.c
        L.minimize = False
        self.assertEqual( blp.check_opposite_objectives(U,L), True )


if __name__ == "__main__":
    unittest.main()
