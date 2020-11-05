import math
import pprint
from scipy.sparse import csr_matrix
import numpy as np
import copy
import collections.abc


class SimplifiedList(collections.abc.MutableSequence):
    """
    This is a normal list class, except if the user asks to
    get attributes, then they will be collected from the
    first list element.  However, in that case, an error
    is generated if there are more than one list elements.
    """

    def __init__(self, clone=None):
        self._clone = clone
        self._data = []

    def clone(self):
        ans = SimplifiedList(clone=self._clone)
        try:
            ans._data = [ val.clone() for val in self._data ]
        except:
            ans._data = [ copy.copy(val) for val in self._data ]
        return ans

    def insert(self, i, val):
        self._data.insert(i, val)

    def __iter__(self):
        for v in self._data:
            yield v

    def __getitem__(self, i):
        if i >= len(self._data):
            if self._clone is None:
                raise IndexError
            for i in range(len(self._data), i+1):
                self.insert(i, copy.copy(self._clone))
        return self._data[i]

    def __setitem__(self, i, val):
        self._data[i] = val

    def __delitem__(self, i):
        del self._data[i]

    def __len__(self):
        return len(self._data)

    def __getattr__(self, name):
        if not name.startswith('_'):
            assert (len(self._data) == 1), "Getting attributes of a simplified list, which has %d elements" % len(self._data)
            return getattr(self._data[0], name)
        return super().__getattr__(name)

    def __setattr__(self, name, val):
        if not name.startswith('_'):
            assert (len(self._data) == 1), "Setting attributes of a simplified list, which has %d elements" % len(self._data)
            return setattr(self._data[0], name, val)
        return super().__setattr__(name, val)


class LevelVariable(object):

    def __init__(self, nxR=0, nxZ=0, nxB=0, lb=None, ub=None):
        self.nxR = nxR
        self.nxZ = nxZ
        self.nxB = nxB
        self.num = nxR+nxZ+nxB
        self.values = [None]*self.num
        if lb is None:
            self.lower_bounds = np.array([np.NINF]*self.num)
        else:
            self.lower_bounds = lb
        if ub is None:
            self.upper_bounds = np.array([np.PINF]*self.num)
        else:
            self.upper_bounds = ub
        for i in range(nxB):
            self.lower_bounds[i+nxR+nxZ] = 0
            self.upper_bounds[i+nxR+nxZ] = 1

    def clone(self):
        ans = LevelVariable(self.nxR, self.nxZ, self.nxB, self.lower_bounds, self.upper_bounds)
        ans.values = copy.copy(self.values)
        return ans

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield i

    def resize(self, nxR, nxZ, nxB, lb=np.NINF, ub=np.PINF):
        if nxR == self.nxR and nxZ == self.nxR and nxB == self.nxB:
            return

        num = nxR+nxZ+nxB
        curr_num = self.num
        self.num = num
        self.values = [None]*num

        tlower = self.lower_bounds
        tupper = self.upper_bounds
        self.lower_bounds = np.array([lb]*num, dtype=np.float64)
        self.upper_bounds = np.array([ub]*num, dtype=np.float64)

        for i in range(min(nxR,self.nxR)):
            self.lower_bounds[i] = tlower[i]
            self.upper_bounds[i] = tupper[i]
        for i in range(min(nxZ,self.nxZ)):
            self.lower_bounds[i+nxR] = tlower[i+self.nxR]
            self.upper_bounds[i+nxR] = tupper[i+self.nxR]
        for i in range(min(nxB,self.nxB)):
            self.lower_bounds[i+nxR+nxZ] = tlower[i+self.nxR+self.nxZ]
            self.upper_bounds[i+nxR+nxZ] = tupper[i+self.nxR+self.nxZ]

        self.nxR = nxR
        self.nxZ = nxZ
        self.nxB = nxB

    def print(self, vtype):                  # pragma: no cover
        print("  %s Variables:" % vtype)
        print("    num: "+str(self.num))
        if self.lower_bounds is not None:
            print("    lower bounds: "+str(self.lower_bounds))
        if self.upper_bounds is not None:
            print("    upper bounds: "+str(self.upper_bounds))
        print("    nonzero values:")
        for i,v in enumerate(self.values):
            if v is not None and v != 0:
                if type(v) is int:
                    print("      %d: %d" % (i, v))
                else:
                    print("      %d: %f" % (i, v))

    def __setattr__(self, name, value):
        if name == 'lower_bounds' and value is not None:
            # Add this check in the model checks
            assert (len(value) == self.num), "The variable has length %s but specifying a lower bounds with length %s" % (str(self.num), str(len(value)))
            if type(value) is list:
                value = np.array(value, dtype=np.float64)
            #print(value)
            #print(value.dtype)
            super().__setattr__(name, value)
        elif name == 'upper_bounds' and value is not None:
            # Add this check in the model checks
            assert (len(value) == self.num), "The variable has length %s but specifying a upper bounds with length %s" % (str(self.num), str(len(value)))
            if type(value) is list:
                value = np.array(value, dtype=float)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class LevelValues(object):

    def __init__(self, matrix=False):
        self._matrix = matrix
        self.x = None

    def clone(self):
        ans = LevelValues(matrix=self._matrix)
        if self._matrix:
            if self.x is not None:
                ans.x = self.x.copy()
        else:
            if self.x is not None:
                ans.x = np.copy(self.x)
        return ans

    def set_values(self, x=None):
        self.x = x

    def print(self, prefix):                        # pragma: no cover
        self._print_value(self.x, prefix+'.x')

    def __len__(self):
        n = 0
        if self._matrix:
            if self.x is not None:
                n = max(n, self.x.shape[0])
        else:
            if self.x is not None:
                n += self.x.size
        return n

    def __setattr__(self, name, value):
        if name in ['x'] and value is not None:
            if type(value) is list:
                if self._matrix:                
                    value = csr_matrix( value )
                else:
                    value = np.array(value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def _print_value(self, value, name):            # pragma: no cover
        if value is not None:
            if self._matrix:
                print("    %s:" % name)
                if value.size:
                    print("        shape: %d %d" % (value.shape[0], value.shape[1]))
                print("        nonzeros:")
                for row in str(value).split('\n'):
                    print("        "+row)
            else:
                print("    %s:" % name, value)


class LevelValueWrapper(object):

    def __init__(self, prefix, matrix=False):
        setattr(self, '_matrix', matrix)
        setattr(self, '_values', {})
        setattr(self, '_prefix', prefix)

    def clone(self):
        ans = LevelValueWrapper(self._prefix, matrix=self._matrix)
        for name in self._values:
            ans._values[name] = self._values[name].clone()
        return ans

    def __len__(self):
        _values = getattr(self, '_values')
        n = 0
        for val in _values.values():
            n += len(val)
        return n

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        else:
            _values = getattr(self, '_values')
            if name in _values:
                return _values[name]
            if name == 'L':
                _values[name] = SimplifiedList(clone=LevelValues(self._matrix))
                _values[name].append(LevelValues(self._matrix))
            else:
                _values[name] = LevelValues(self._matrix)
            return _values[name]

    def print(self, *args, nL=0):               # pragma: no cover
        _values = getattr(self, '_values')
        first = True
        for name in args:
            v = _values.get(name,None)
            if v is None:
                continue
            if name == 'L':
                if nL == 1:
                    if len(v) > 0:
                        if first:
                            print("  "+self._prefix+":")
                            first = False
                        v.print(name)
                else:
                    for i,L in enumerate(v):
                        if len(L) > 0:
                            if first:
                                print("  "+self._prefix+":")
                                first = False
                            L.print(name+"[%d]"%i)
            else:
                if len(v) > 0:
                    if first:
                        print("  "+self._prefix+":")
                        first = False
                    v.print(name)
            

class LinearLevelRepn(object):

    def __init__(self, nxR, nxZ, nxB):
        self.x = LevelVariable(nxR, nxZ, nxB)    # variables at this level
        self.c = LevelValueWrapper("c") # objective coefficients at this level
        self.A = LevelValueWrapper("A",
                        matrix=True)    # constraint matrices at this level
        self.b = np.ndarray(0)          # RHS of the constraints
        self.minimize = True            # sense of the objective at this level
        self.inequalities = True        # If True, the constraints are inequalities
        self.d = 0                      # constant in objective at this level

    def clone(self):
        ans = LinearLevelRepn(0,0,0)
        ans.x = self.x.clone()
        ans.c = self.c.clone()
        ans.A = self.A.clone()
        ans.b = np.copy(self.b)
        ans.minimize = self.minimize
        ans.inequalities = self.inequalities
        ans.d = self.d
        # TODO - Should we allow users to annotate these objects with other data?
        for attr in dir(self):
            if attr in ['x', 'c', 'A', 'b', 'minimize', 'inequalities', 'd']:
                continue
            if attr in ['clone', 'print']:    # methods
                continue
            if attr.startswith('_'):
                continue
            setattr(ans, attr, copy.copy(getattr(self, attr)))
        return ans

    def print(self, *args, nL=0):       # pragma: no cover
        print("Variables:")
        self.x.print()

        print("\nObjective:")
        if self.minimize:
            print("  Minimize:")
        else:
            print("  Maximize:")
        self.c.print(*args, nL=nL)
        print("  d:",self.d)

        if self.b.size > 0:
            print("\nConstraints: ")
            self.A.print(*args, nL=nL)
            if self.inequalities:
                print("  <=", self.b)
            else:
                print("  ==", self.b)

    def __setattr__(self, name, value):
        if name == 'b' and value is not None:
            value = np.array(value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class LinearBilevelProblem(object):
    """
    Let
        x   = [U.x, L.x]'                         # dense column vector
        U.c = [U.c.U.x, U.c.L.x]  # dense row vector
        L.c = [L.c.U.x, L.c.L.x]  # dense row vector
        U.A = [U.A.U.x, U.A.L.x]  # sparse matrix
        L.A = [L.A.U.x, L.A.L.x]  # sparse matrix

    min_{U.x}   U.c * x + U.d
    s.t.        U.A * x <= U.b                      # Or ==

                where L.x satisifies

                    min_{L.x}   L.c * x + L.d
                    s.t.        L.A * x <= L.b      # Or ==
    """

    def __init__(self, name=None):
        self.name = name
        #self.model = None
        self.U = None
        self.L = SimplifiedList()

    def add_upper(self, *, nxR=0, nxZ=0, nxB=0):
        assert (self.U is None), "Cannot create a second upper-level in a LinearBilevelProblem"
        self.U = LinearLevelRepn(nxR, nxZ, nxB)
        return self.U

    def add_lower(self, *, nxR=0, nxZ=0, nxB=0):
        self.L.append( LinearLevelRepn(nxR, nxZ, nxB) )
        return self.L

    def clone(self):
        ans = LinearBilevelProblem()
        ans.name = self.name
        ans.U = self.U.clone()
        ans.L = self.L.clone()
        return ans

    def print(self):                            # pragma: no cover
        nL = len(self.L)
        if self.name:
            print("# LinearBilevelProblem: "+self.name)
        else:
            print("# LinearBilevelProblem: unknown")
        print("")
        print("## Upper Level")
        print("")
        self.U.print("U", "L", nL=nL)
        print("")

        if len(self.L) == 1:
            print("## Lower Level")
            print("")
            self.L.print("U", "L", nL=nL)
        else:
            for i,L in enumerate(self.L):
                print("## Lower Level: "+str(i))
                print("")
                L.print("U", "L", nL=nL)
                print("")

    def check(self):                    # pragma: no cover
        U = self.U
        L = self.L
        #
        # Coefficients for upper-level objective
        #
        assert ((U.c.U.x is None) or (U.c.U.x.size == len(U.x)) or (U.c.U.x.size == 0)), "Incompatible specification of upper-level coefficients for U.x"
        for i in range(len(L)):
            assert ((U.c.L[i].x is None) or (U.c.L[i].x.size == len(L[i].x)) or (U.c.L[i].x.size == 0)), "Incompatible specification of upper-level coefficients for L[%d].x" % i
        #
        # Coefficients for lower-level objective
        #
        for i in range(len(L)):
            assert ((L[i].c.U.x is None) or (L[i].c.U.x.size == len(U.x)) or (L[i].c.U.x.size == 0)), "Incompatible specification of lower-level coefficients for U.x" 
            assert ((L[i].c.L[i].x is None) or (L[i].c.L[i].x.size == len(L[i].x)) or (L[i].c.L[i].x.size == 0)), "Incompatible specification of lower-level coefficients for L[%d].x" % i
        #
        # Ncols of upper-level constraints
        #
        assert ((U.A.U.x is None) or (U.A.U.x.shape[1] == len(U.x))), "Incompatible specification of U.A.U.x and U.x"
        for i in range(len(L)):
            assert ((L[i].A.U.x is None) or (L[i].A.U.x.shape[1] == len(U.x))), "Incompatible specification of L[%d].A.U.x and U.x" % i
        #
        # Ncols of lower-level constraints
        #
        for i in range(len(L)):
            assert ((U.A.L[i].x is None) or (U.A.L[i].x.shape[1] == len(L[i].x))), "Incompatible specification of U.A.L[%d].x and L[%d].x" % (i,i)
            assert ((L[i].A.L[i].x is None) or (L[i].A.L[i].x.shape[1] == len(L[i].x))), "Incompatible specification of L[%d].A.L[%d].x and L[%d].x" % (i,i,i)
        #
        # Nrows of upper-level constraints
        #
        if U.b is None:
            assert (U.A.U.x is None), "Incompatible specification of U.b and U.A.U.x"
            for i in range(len(L)):
                assert (U.A.L[i].x is None), "Incompatible specification of U.b and U.A.L[%d].x" % i
        else:
            nr = U.b.size
            if U.A.U.x is not None:
                if nr != U.A.U.x.shape[0]:
                    print("X", nr, U.A.U.x.shape[0])
                assert (nr == U.A.U.x.shape[0]), "Incompatible specification of U.b and U.A.U.x"
            for i in range(len(L)):
                if U.A.L[i].x is not None:
                    assert (nr == U.A.L[i].x.shape[0]), "Incompatible specification of U.b and U.A.L[%d].x" % i
        #
        # Nrows of lower-level constraints
        #
        for i in range(len(L)):
            if L[i].b is None:
                assert (L[i].A.U.x is None), "Incompatible specification of L[%d].b and L[%d].A.U.x" % (i,i)
                assert (L[i].A.L[i].x is None), "Incompatible specification of L[%d].b and L[%d].A.L[%d].x" % (i,i,i)
            else:
                nr = L[i].b.size
                if L[i].A.U.x is not None:
                    assert (nr == L[i].A.U.x.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.U.x" % (i,i)
                if L[i].A.L[i].x is not None:
                    assert (nr == L[i].A.L[i].x.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.L[%d].x" % (i,i,i)

    def check_opposite_objectives(self, U, L):
        if id(U.c) == id(L.c) and L.minimize ^ U.minimize:
            return True
        U_coef = 1 if U.minimize else -1
        L_coef = 1 if L.minimize else -1
        if not self._equal_nparray(U.c.U.x, U_coef, L.c.U.x, L_coef):
            return False
        if not self._equal_nparray(U.c.L.x, U_coef, L.c.L.x, L_coef):
            return False
        return True

    def _equal_nparray(self, Ux, U_coef, Lx, L_coef):
        if Ux is None and Lx is None:
            return True
        if Ux is None or Lx is None:
            return False
        for i in range(Ux.size):
            if math.fabs(Ux[i]*U_coef + Lx[i]*L_coef) > 1e-16:
                return False
        return True


if __name__ == "__main__":              # pragma: no cover
    prob = LinearBilevelProblem()
    U = prob.add_upper(3,2,1)
    U.x.upper_bounds = np.array([1.5, 2.4, 3.1])
    L = prob.add_lower(1,2,3)
    L.xZ.lower_bounds = np.array([1, -2])
    prob.print()

