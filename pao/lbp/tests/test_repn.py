import numpy as np
import scipy.sparse
import pyutilib.th as unittest
from pao.lbp import *
from pao.lbp.repn import SimplifiedList, LinearLevelRepn, LevelVariable, LevelValues, LevelValueWrapper


class Test_SimplifiedList(unittest.TestCase):

    def test_init(self):
        l = SimplifiedList()
        self.assertEqual(len(l._data),0)

    def test_clone1(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
        l = SimplifiedList()
        l.append(A())
        l.insert(2, A(2))
        ans = l.clone()
        self.assertEqual(len(ans), 2)
        
    def test_clone2(self):
        class A(object):
            def __init__(self, i=0):
                self.i = i
            def clone(self):
                return A(self.i+1)
        l = SimplifiedList()
        l.append(A())
        l.insert(2, A(2))
        ans = l.clone()
        self.assertEqual(len(ans), 2)
        self.assertEqual(ans[-1].i, 3)
        
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

        l = LevelVariable(3)
        l.lower_bounds = [1,2,3]
        l.resize(3, lb=np.NINF)
        self.assertEqual(list(l.lower_bounds), [1,2,3])

        l = LevelVariable(3)
        l.lower_bounds = [1,2,3]
        l.upper_bounds = [4,5,6]
        l.resize(2, lb=4)
        self.assertEqual(list(l.lower_bounds), [1,2])
        self.assertEqual(list(l.upper_bounds), [4,5])

        l = LevelVariable(3)
        l.lower_bounds = [1,2,3]
        l.upper_bounds = [3,4,5]
        l.resize(4, lb=4, ub=6)
        self.assertEqual(list(l.lower_bounds), [1,2,3,4])
        self.assertEqual(list(l.upper_bounds), [3,4,5,6])

        l = LevelVariable(3)
        l.resize(4, lb=4)
        self.assertEqual(list(l.lower_bounds), [np.NINF,np.NINF,np.NINF,4])
        self.assertEqual(list(l.upper_bounds), [np.PINF,np.PINF,np.PINF,np.PINF])

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
        l.set_values(xR=[[0,2,0,0,0],[0,0,0,0,0],[0,0,0,0,5]], xB=[[5],[6]], xZ=[[0,0],[0,1]])
        self.assertEqual(len(l), 3)
        l.set_values(xR=[[0,2,0,0,0],[0,0,0,0,0],[0,0,0,0,5]], xB=[[5],[6]], xZ=[[0,0],[0,1]])
        self.assertEqual(len(l), 3)
        
    def test_clone_values(self):
        l = LevelValues()
        l.set_values(xR=[0,1,2], xZ=[3,4], xB=[5])
        ans = l.clone()
        self.assertEqual(len(ans), 6)
        
    def test_clone_matrix(self):
        l = LevelValues(matrix=True)
        l.set_values(xR=[[0,2,0,0,0],[0,0,0,0,0],[0,0,0,0,5]], xB=[[5],[6]], xZ=[[0,0],[0,1]])
        ans = l.clone()
        self.assertEqual(len(ans), 3)
        l.set_values(xR=[[0,2,0,0,0],[0,0,0,0,0],[0,0,0,0,5]], xB=[[5],[6]], xZ=[[0,0],[0,1]])
        ans = l.clone()
        self.assertEqual(len(ans), 3)
        
    def test_setattr(self):
        l = LevelValues()
        l.a = [1,2,3]
        self.assertEqual(type(l.a),list)
        l.xR = [1,2,3]
        self.assertEqual(type(l.xR),np.ndarray)

    def test_setattr_matrix(self):
        l = LevelValues(matrix=True)
        l.a = [[1,0,2,0],
               [0,3,0,4]]
        self.assertEqual(type(l.a),list)
        l.xR = [[1,0,2,0],
                [0,3,0,4]]
        self.assertEqual(type(l.xR),scipy.sparse.csr.csr_matrix)


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
        
    def test_clone(self):
        l = LevelValueWrapper('foo', matrix=True)
        l.foo
        l.L
        ans = l.clone()
        self.assertEqual(list(ans._values.keys()), ['foo','L'])
        self.assertEqual(type(ans.foo), LevelValues)
        self.assertEqual(len(ans.L),1)
        

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

    def test_clone(self):
        l = LinearLevelRepn(1,2,3)
        l.x = -1
        l.b = -1
        ans = l.clone()
        self.assertEqual(ans.x, -1)
        self.assertEqual(ans.b, np.array(-1))
        

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

    def test_clone(self):
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=1, nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)

        ans = blp.clone()
        self.assertEqual(len(ans.U.xR), 1)
        self.assertEqual(len(ans.U.xZ), 2)
        self.assertEqual(len(ans.U.xB), 3)

        self.assertEqual(len(ans.L[0].xR), 1)
        self.assertEqual(len(ans.L[0].xZ), 2)
        self.assertEqual(len(ans.L[0].xB), 3)

        self.assertEqual(len(ans.L[1].xR), 1)
        self.assertEqual(len(ans.L[1].xZ), 2)
        self.assertEqual(len(ans.L[1].xB), 3)

    def test_check_matrix_initialization(self):
        blp = LinearBilevelProblem()
        U = blp.add_upper(nxR=2, nxZ=3, nxB=4)
        L = blp.add_lower(nxR=1, nxZ=2, nxB=3)

        U.b = [1,2,3]

        U.A.U.xR = [[1,0],[0,0],[0,0]]
        U.A.U.xZ = [[1,0,0],[0,0,0],[0,0,0]]
        U.A.U.xB = [[1,0,0,0],[0,0,0,0],[0,0,0,0]]
        U.A.L.xR = [[1],[0],[0]]
        U.A.L.xZ = [[1,0],[0,0],[0,0]]
        U.A.L.xB = [[1,0,0],[0,0,0],[0,0,0]]

        U.L = [1,2,3,4]

        L.A.U.xR = [[1,0],[0,0],[0,0]]
        L.A.U.xZ = [[1,0,0],[0,0,0],[0,0,0]]
        L.A.U.xB = [[1,0,0,0],[0,0,0,0],[0,0,0,0]]
        L.A.L.xR = [[1],[0],[0]]
        L.A.L.xZ = [[1,0],[0,0],[0,0]]
        L.A.L.xB = [[1,0,0],[0,0,0],[0,0,0]]

        self.assertEqual(U.A.U.xR.shape, (3,2))
        self.assertEqual(U.A.U.xZ.shape, (3,3))
        self.assertEqual(U.A.U.xB.shape, (3,4))

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
