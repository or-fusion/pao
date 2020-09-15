import math
import pprint
from scipy.sparse import coo_matrix
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

    def __init__(self, num, lb=None, ub=None):
        self.num = num
        self.values = [None]*num
        self.lower_bounds = lb
        self.upper_bounds = ub

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield i

    def resize(self, num):
        if self.num != num:
            self.lower_bounds = None
            self.upper_bounds = None
        self.num = num
        self.values = [None]*num

    def print(self, type):                  # pragma: no cover
        print("  %s Variables:" % type)
        print("    num: "+str(self.num))
        if self.lower_bounds is not None:
            print("    lower bounds: "+str(self.lower_bounds))
        if self.upper_bounds is not None:
            print("    upper bounds: "+str(self.upper_bounds))
        print("    nonzero values:")
        for i,v in enumerate(self.values):
            if v is not None and v != 0:
                print("      %d: %f" % (i, v))

    def __setattr__(self, name, value):
        if name == 'lower_bounds' and value is not None:
            # Add this check in the model checks
            #assert (value.size == self.num), "The variable has length %s but specifying a lower bounds with length %s" % (str(self.num), str(value.size))
            if type(value) is list:
                value = np.array(value)
            super().__setattr__(name, value)
        elif name == 'upper_bounds' and value is not None:
            # Add this check in the model checks
            #assert (value.size == self.num), "The variable has length %s but specifying a upper bounds with length %s" % (str(self.num), str(value.size))
            if type(value) is list:
                value = np.array(value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class LevelValues(object):

    def __init__(self, matrix=False):
        self._matrix = matrix
        self.xR = None
        self.xZ = None
        self.xB = None

    def set_values(self, xB=None, xR=None, xZ=None):
        self.xR = xR
        self.xB = xB
        self.xZ = xZ

    def print(self, prefix):                        # pragma: no cover
        self._print_value(self.xR, prefix+'.xR')
        self._print_value(self.xZ, prefix+'.xZ')
        self._print_value(self.xB, prefix+'.xB')

    def __len__(self):
        n = 0
        if self._matrix:
            if self.xR is not None:
                n = max(n, self.xR.shape[0])
            if self.xZ is not None:
                n = max(n, self.xZ.shape[0])
            if self.xB is not None:
                n = max(n, self.xB.shape[0])
        else:
            if self.xR is not None:
                n += self.xR.size
            if self.xZ is not None:
                n += self.xZ.size
            if self.xB is not None:
                n += self.xB.size
        return n

    def __setattr__(self, name, value):
        if name in ['xR', 'xZ', 'xB'] and value is not None:
            if type(value) is list:
                if self._matrix:                
                    ivals = []
                    jvals = []
                    vals = []
                    for i,j,v in value:
                        ivals.append(i)
                        jvals.append(j)
                        vals.append(v)
                    value = coo_matrix( (vals, (ivals,jvals)) )
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
                print("        nonzeors:")
                for row in str(value).split('\n'):
                    print("        "+row)
            else:
                print("    %s:" % name, value)


class LevelValueWrapper(object):

    def __init__(self, prefix, matrix=False):
        setattr(self, '_matrix', matrix)
        setattr(self, '_values', {})
        setattr(self, '_prefix', prefix)

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
        self.xR = LevelVariable(nxR)    # continuous variables at this level
        self.xZ = LevelVariable(nxZ)    # integer variables at this level
        self.xB = LevelVariable(nxB)    # binary variables at this level
        self.minimize = True            # sense of the objective at this level
        self.c = LevelValueWrapper("c") # objective coefficients at this level
        self.d = 0                      # constant in objective at this level
        self.A = LevelValueWrapper("A",
                        matrix=True)    # constraint matrices at this level
        self.b = np.ndarray(0)          # RHS of the constraints
        self.inequalities = True        # If True, the constraints are inequalities

    def print(self, *args, nL=0):       # pragma: no cover
        print("Variables:")
        if self.xR.num > 0:
            self.xR.print("Real")
        if self.xZ.num > 0:
            self.xZ.print("Integer")
        if self.xB.num > 0:
            self.xB.print("Binary")

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
        x   = [U.xR, U.xZ, U.xB, L.xR, L.xZ, L.xB]'                         # dense column vector
        U.c = [U.c.U.xR, U.c.U.xZ, U.c.U.xB, U.c.L.xR, U.c.L.xZ, U.c.L.xB]  # dense row vector
        L.c = [L.c.U.xR, L.c.U.xZ, L.c.U.xB, L.c.L.xR, L.c.L.xZ, L.c.L.xB]  # dense row vector
        U.A = [U.A.U.xR, U.A.U.xZ, U.A.U.xB, U.A.L.xR, U.A.L.xZ, U.A.L.xB]  # sparse matrix
        L.A = [L.A.U.xR, L.A.U.xZ, L.A.U.xB, L.A.L.xR, L.A.L.xZ, L.A.L.xB]  # sparse matrix

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

    def print(self):                            # pragma: no cover
        nL = len(self.L)
        if self.name:
            print("# LinearBilevelProblem: "+name)
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
        #print(U.c.U.xR.size, U.xR.num, U.xR.values)
        assert ((U.c.U.xR is None) or (U.c.U.xR.size == len(U.xR)) or (U.c.U.xR.size == 0)), "Incompatible specification of upper-level coefficients for U.xR"
        assert ((U.c.U.xZ is None) or (U.c.U.xZ.size == len(U.xZ)) or (U.c.U.xZ.size == 0)), "Incompatible specification of upper-level coefficients for U.xZ"
        assert ((U.c.U.xB is None) or (U.c.U.xB.size == len(U.xB)) or (U.c.U.xB.size == 0)), "Incompatible specification of upper-level coefficients for U.xB"
        for i in range(len(L)):
            assert ((U.c.L[i].xR is None) or (U.c.L[i].xR.size == len(L[i].xR)) or (U.c.L[i].xR.size == 0)), "Incompatible specification of upper-level coefficients for L[%d].xR" % i
            assert ((U.c.L[i].xZ is None) or (U.c.L[i].xZ.size == len(L[i].xZ)) or (U.c.L[i].xZ.size == 0)), "Incompatible specification of upper-level coefficients for L[%d].xZ" % i
            assert ((U.c.L[i].xB is None) or (U.c.L[i].xB.size == len(L[i].xB)) or (U.c.L[i].xB.size == 0)), "Incompatible specification of upper-level coefficients for L[%d].xB" % i
        #
        # Coefficients for lower-level objective
        #
        for i in range(len(L)):
            #print("HERE", i, L[i].c.U.xR.size, len(U.xR))
            assert ((L[i].c.U.xR is None) or (L[i].c.U.xR.size == len(U.xR)) or (L[i].c.U.xR.size == 0)), "Incompatible specification of lower-level coefficients for U.xR" 
            assert ((L[i].c.U.xZ is None) or (L[i].c.U.xZ.size == len(U.xZ)) or (L[i].c.U.xZ.size == 0)), "Incompatible specification of lower-level coefficients for U.xZ"
            assert ((L[i].c.U.xB is None) or (L[i].c.U.xB.size == len(U.xB)) or (L[i].c.U.xB.size == 0)), "Incompatible specification of lower-level coefficients for U.xB"
            assert ((L[i].c.L[i].xR is None) or (L[i].c.L[i].xR.size == len(L[i].xR)) or (L[i].c.L[i].xR.size == 0)), "Incompatible specification of lower-level coefficients for L[%d].xR" % i
            assert ((L[i].c.L[i].xZ is None) or (L[i].c.L[i].xZ.size == len(L[i].xZ)) or (L[i].c.L[i].xZ.size == 0)), "Incompatible specification of lower-level coefficients for L[%d].xZ" % i
            assert ((L[i].c.L[i].xB is None) or (L[i].c.L[i].xB.size == len(L[i].xB)) or (L[i].c.L[i].xB.size == 0)), "Incompatible specification of lower-level coefficients for L[%d].xB" % i
        #
        # Ncols of upper-level constraints
        #
        assert ((U.A.U.xR is None) or (U.A.U.xR.shape[1] == len(U.xR)) or len(U.xR) == 0), "Incompatible specification of U.A.U.xR and U.xR"
        assert ((U.A.U.xZ is None) or (U.A.U.xZ.shape[1] == len(U.xZ)) or len(U.xZ) == 0), "Incompatible specification of U.A.U.xZ and U.xZ"
        assert ((U.A.U.xB is None) or (U.A.U.xB.shape[1] == len(U.xB)) or len(U.xB) == 0), "Incompatible specification of U.A.U.xB and U.xB"
        for i in range(len(L)):
            assert ((L[i].A.U.xR is None) or (L[i].A.U.xR.shape[1] == len(U.xR)) or len(U.xR) == 0), "Incompatible specification of L[%d].A.U.xR and U.xR" % i
            assert ((L[i].A.U.xZ is None) or (L[i].A.U.xZ.shape[1] == len(U.xZ)) or len(U.xZ) == 0), "Incompatible specification of L[%d].A.U.xZ and U.xZ" % i
            assert ((L[i].A.U.xB is None) or (L[i].A.U.xB.shape[1] == len(U.xB)) or len(U.xB) == 0), "Incompatible specification of L[%d].A.U.xB and U.xB" % i
        #
        # Ncols of lower-level constraints
        #
        for i in range(len(L)):
            assert ((U.A.L[i].xR is None) or (U.A.L[i].xR.shape[1] == len(L[i].xR)) or len(L[i].xR) == 0), "Incompatible specification of U.A.L[%d].xR and L[%d].xR" % (i,i)
            assert ((U.A.L[i].xZ is None) or (U.A.L[i].xZ.shape[1] == len(L[i].xZ)) or len(L[i].xZ) == 0), "Incompatible specification of U.A.L[%d].xZ and L[%d].xZ" % (i,i)
            assert ((U.A.L[i].xB is None) or (U.A.L[i].xB.shape[1] == len(L[i].xB)) or len(L[i].xB) == 0), "Incompatible specification of U.A.L[%d].xB and L[%d].xB" % (i,i)
            assert ((L[i].A.L[i].xR is None) or (L[i].A.L[i].xR.shape[1] == len(L[i].xR)) or len(L[i].xR) == 0), "Incompatible specification of L[%d].A.L[%d].xR and L[%d].xR" % (i,i,i)
            assert ((L[i].A.L[i].xZ is None) or (L[i].A.L[i].xZ.shape[1] == len(L[i].xZ)) or len(L[i].xZ) == 0), "Incompatible specification of L[%d].A.L[%d].xZ and L[%d].xZ" % (i,i,i)
            assert ((L[i].A.L[i].xB is None) or (L[i].A.L[i].xB.shape[1] == len(L[i].xB)) or len(L[i].xB) == 0), "Incompatible specification of L[%d].A.L[%d].xB and L[%d].xB" % (i,i,i)
        #
        # Nrows of upper-level constraints
        #
        if U.b is None:
            assert (U.A.U.xR is None), "Incompatible specification of U.b and U.A.U.xR"
            assert (U.A.U.xZ is None), "Incompatible specification of U.b and U.A.U.xZ"
            assert (U.A.U.xB is None), "Incompatible specification of U.b and U.A.U.xB"
            for i in range(len(L)):
                assert (U.A.L[i].xR is None), "Incompatible specification of U.b and U.A.L[%d].xR" % i
                assert (U.A.L[i].xZ is None), "Incompatible specification of U.b and U.A.L[%d].xZ" % i
                assert (U.A.L[i].xB is None), "Incompatible specification of U.b and U.A.L[%d].xB" % i
        else:
            nr = U.b.size
            if U.A.U.xR is not None:
                assert (nr == U.A.U.xR.shape[0]), "Incompatible specification of U.b and U.A.U.xR"
            if U.A.U.xZ is not None:
                assert (nr == U.A.U.xZ.shape[0]), "Incompatible specification of U.b and U.A.U.xZ"
            if U.A.U.xB is not None:
                assert (nr == U.A.U.xB.shape[0]), "Incompatible specification of U.b and U.A.U.xB"
            for i in range(len(L)):
                if U.A.L[i].xR is not None:
                    assert (nr == U.A.L[i].xR.shape[0]), "Incompatible specification of U.b and U.A.L[%d].xR" % i
                if U.A.L[i].xZ is not None:
                    assert (nr == U.A.L[i].xZ.shape[0]), "Incompatible specification of U.b and U.A.L[%d].xZ" % i
                if U.A.L[i].xB is not None:
                    assert (nr == U.A.L[i].xB.shape[0]), "Incompatible specification of U.b and U.A.L[%d].xB" % i
        #
        # Nrows of lower-level constraints
        #
        for i in range(len(L)):
            if L[i].b is None:
                assert (L[i].A.U.xR is None), "Incompatible specification of L[%d].b and L[%d].A.U.xR" % (i,i)
                assert (L[i].A.U.xZ is None), "Incompatible specification of L[%d].b and L[%d].A.U.xZ" % (i,i)
                assert (L[i].A.U.xB is None), "Incompatible specification of L[%d].b and L[%d].A.U.xB" % (i,i)
                assert (L[i].A.L[i].xR is None), "Incompatible specification of L[%d].b and L[%d].A.L[%d].xR" % (i,i,i)
                assert (L[i].A.L[i].xZ is None), "Incompatible specification of L[%d].b and L[%d].A.L[%d].xZ" % (i,i,i)
                assert (L[i].A.L[i].xB is None), "Incompatible specification of L[%d].b and L[%d].A.L[%d].xB" % (i,i,i)
            else:
                nr = L[i].b.size
                if L[i].A.U.xR is not None:
                    assert (nr == L[i].A.U.xR.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.U.xR" % (i,i)
                if L[i].A.U.xZ is not None:
                    assert (nr == L[i].A.U.xZ.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.U.xZ" % (i,i)
                if L[i].A.U.xB is not None:
                    assert (nr == L[i].A.U.xB.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.U.xB" % (i,i)
                if L[i].A.L[i].xR is not None:
                    assert (nr == L[i].A.L[i].xR.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.L[%d].xR" % (i,i,i)
                if L[i].A.L[i].xZ is not None:
                    assert (nr == L[i].A.L[i].xZ.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.L[%d].xZ" % (i,i,i)
                if L[i].A.L[i].xB is not None:
                    assert (nr == L[i].A.L[i].xB.shape[0]), "Incompatible specification of L[%d].b and L[%d].A.L[%d].xB" % (i,i,i)

    def check_opposite_objectives(self, U, L):
        if id(U.c) == id(L.c) and L.minimize ^ U.minimize:
            return True
        U_coef = 1 if U.minimize else -1
        L_coef = 1 if L.minimize else -1
        if not self._equal_nparray(U.c.U.xR, U_coef, L.c.U.xR, L_coef):
            return False
        if not self._equal_nparray(U.c.U.xZ, U_coef, L.c.U.xZ, L_coef):
            return False
        if not self._equal_nparray(U.c.U.xB, U_coef, L.c.U.xB, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xR, U_coef, L.c.L.xR, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xZ, U_coef, L.c.L.xZ, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xB, U_coef, L.c.L.xB, L_coef):
            return False
        return True

    def _equal_nparray(self, U, U_coef, L, L_coef):
        if U is None and L is None:
            return True
        if U is None or L is None:
            return False
        for i in range(U.size):
            if math.fabs(U[i]*U_coef + L[i]*L_coef) > 1e-16:
                return False
        return True


class QuadraticLevelRepn(object):       # pragma: no cover

    def __init__(self, nxR, nxZ, nxB):
        super().__init__(nxR, nxZ, nxB)
        self.B = LevelValueWrapper("B") # Quadratic term in objective

    def print(self, *args, nL=0):
        print("Variables:")
        if self.xR.num > 0:
            self.xR.print("Real")
        if self.xZ.num > 0:
            self.xZ.print("Integer")
        if self.xB.num > 0:
            self.xB.print("Binary")

        print("\nObjective:")
        if self.minimize:
            print("  Minimize:")
        else:
            print("  Maximize:")
        self.c.print(*args, nL=nL)
        self.B.print(*args)

        if len(self.A) > 0:
            print("\nConstraints:")
            self.A.print(*args)
            if self.inequalities:
                print("  <=")
            else:
                print("  ==")
            print("   ",self.b)


class QuadraticBilevelProblem(LinearBilevelProblem):            # pragma: no cover
    """
    Let
        x   = [U.xR, U.xZ, U.xB, L.xR, L.xZ, L.xB]'                         # dense column vector
        U.x = [U.xR, U.xZ, U.xB]'                                           # dense column vector
        L.x = [L.xR, L.xZ, L.xB]'                                           # dense column vector
        U.c = [U.c.U.xR, U.c.U.xZ, U.c.U.xB, U.c.L.xR, U.c.L.xZ, U.c.L.xB]  # dense row vector
        L.c = [L.c.U.xR, L.c.U.xZ, L.c.U.xB, L.c.L.xR, L.c.L.xZ, L.c.L.xB]  # dense row vector
        U.A = [U.A.U.xR, U.A.U.xZ, U.A.U.xB, U.A.L.xR, U.A.L.xZ, U.A.L.xB]  # sparse matrix
        L.A = [L.A.U.xR, L.A.U.xZ, L.A.U.xB, L.A.L.xR, L.A.L.xZ, L.A.L.xB]  # sparse matrix
        U.B = [ [U.B.xR.xR, U.B.xR.xZ, U.B.xR.xB],
                [U.B.xZ.xR, U.B.xZ.xZ, U.B.xZ.xB],
                [U.B.xB.xR, U.B.xB.xZ, U.B.xB.xB]]                          # sparse matrix U.x rows, L.x cols
        L.B = [ [L.B.xR.xR, L.B.xR.xZ, L.B.xR.xB],
                [L.B.xZ.xR, L.B.xZ.xZ, L.B.xZ.xB],
                [L.B.xB.xR, L.B.xB.xZ, L.B.xB.xB]]                          # sparse matrix U.x rows, L.x cols

    min_{U.x}   U.c * x + U.x' * U.B * L.x
    s.t.        U.A * x <= U.b                      # Or ==

                where L.x satisifies

                    min_{L.x}   L.c * x + U.x' * L.B * L.x
                    s.t.        L.A * x <= L.b      # Or ==
    """

    def __init__(self, name=None):
        super().__init__(name)

    def add_upper(self, nxR=0, nxZ=0, nxB=0):
        self.U = QuadraticLevelRepn(nxR, nxZ, nxB)
        return self.U

    def add_lower(self, nxR=0, nxZ=0, nxB=0):
        self.L = QuadraticLevelRepn(nxR, nxZ, nxB)
        return self.L

    def print(self):
        if self.name:
            print("# QuadraticBilevelProblem: "+name)
        else:
            print("# QuadraticBilevelProblem: unknown")
        print("")
        print("## Upper Level")
        print("")
        self.U.print("U","L")
        print("")
        print("## Lower Level")
        print("")
        self.L.print("U","L")

    def check(self):
        LinearBilevelProblem.check(self)
        #
        U = self.U
        L = self.L
        #
        # Ncols/Nrows of U.B
        #
        assert ((U.B.xR.xR is None) or (U.B.xR.xR.shape[1] == len(L.xR)) or (U.c.xR.xR.shape[1] == 0)), "Incompatible specification of U.B.xR.xR and L.xR"
        assert ((U.B.xR.xR is None) or (U.B.xR.xR.shape[0] == len(U.xR)) or (U.c.xR.xR.shape[0] == 0)), "Incompatible specification of U.B.xR.xR and U.xR"

        assert ((U.B.xR.xZ is None) or (U.B.xR.xZ.shape[1] == len(L.xZ)) or (U.c.xR.xZ.shape[1] == 0)), "Incompatible specification of U.B.xR.xR and L.xZ"
        assert ((U.B.xR.xZ is None) or (U.B.xR.xZ.shape[0] == len(U.xR)) or (U.c.xR.xZ.shape[0] == 0)), "Incompatible specification of U.B.xR.xR and U.xR"

        assert ((U.B.xR.xB is None) or (U.B.xR.xB.shape[1] == len(L.xB)) or (U.c.xR.xB.shape[1] == 0)), "Incompatible specification of U.B.xR.xR and L.xB"
        assert ((U.B.xR.xB is None) or (U.B.xR.xB.shape[0] == len(U.xR)) or (U.c.xR.xB.shape[0] == 0)), "Incompatible specification of U.B.xR.xR and U.xR"


        assert ((U.B.xZ.xR is None) or (U.B.xZ.xR.shape[1] == len(L.xR)) or (U.c.xZ.xR.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xR"
        assert ((U.B.xZ.xR is None) or (U.B.xZ.xR.shape[0] == len(U.xZ)) or (U.c.xZ.xR.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xZ"

        assert ((U.B.xZ.xZ is None) or (U.B.xZ.xZ.shape[1] == len(L.xZ)) or (U.c.xZ.xZ.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xZ"
        assert ((U.B.xZ.xZ is None) or (U.B.xZ.xZ.shape[0] == len(U.xZ)) or (U.c.xZ.xZ.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xZ"

        assert ((U.B.xZ.xB is None) or (U.B.xZ.xB.shape[1] == len(L.xB)) or (U.c.xZ.xB.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xB"
        assert ((U.B.xZ.xB is None) or (U.B.xZ.xB.shape[0] == len(U.xZ)) or (U.c.xZ.xB.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xZ"


        assert ((U.B.xB.xR is None) or (U.B.xB.xR.shape[1] == len(L.xR)) or (U.c.xB.xR.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xR"
        assert ((U.B.xB.xR is None) or (U.B.xB.xR.shape[0] == len(U.xB)) or (U.c.xB.xR.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xB"

        assert ((U.B.xB.xZ is None) or (U.B.xB.xZ.shape[1] == len(L.xZ)) or (U.c.xB.xZ.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xZ"
        assert ((U.B.xB.xZ is None) or (U.B.xB.xZ.shape[0] == len(U.xB)) or (U.c.xB.xZ.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xB"

        assert ((U.B.xB.xB is None) or (U.B.xB.xB.shape[1] == len(L.xB)) or (U.c.xB.xB.shape[1] == 0)), "Incompatible specification of U.B.xB.xR and L.xB"
        assert ((U.B.xB.xB is None) or (U.B.xB.xB.shape[0] == len(U.xB)) or (U.c.xB.xB.shape[0] == 0)), "Incompatible specification of U.B.xB.xR and U.xB"

    def check_opposite_objectives(self, U, L):
        if id(U.c) == id(L.c) and id(U.B) == id(L.B) and L.minimize ^ U.minimize:
            return True
        U_coef = 1 if U.minimize else -1
        L_coef = 1 if L.minimize else -1

        if not self._equal_nparray(U.c.U.xR, U_coef, L.c.U.xR, L_coef):
            return False
        if not self._equal_nparray(U.c.U.xZ, U_coef, L.c.U.xZ, L_coef):
            return False
        if not self._equal_nparray(U.c.U.xB, U_coef, L.c.U.xB, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xR, U_coef, L.c.L.xR, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xZ, U_coef, L.c.L.xZ, L_coef):
            return False
        if not self._equal_nparray(U.c.L.xB, U_coef, L.c.L.xB, L_coef):
            return False

        if not self._equal_mat(U.c.xR.xR, U_coef, L.c.xR.xR, L_coef):
            return False
        if not self._equal_mat(U.c.xR.xZ, U_coef, L.c.xR.xZ, L_coef):
            return False
        if not self._equal_mat(U.c.xR.xB, U_coef, L.c.xR.xB, L_coef):
            return False
        if not self._equal_mat(U.c.xZ.xR, U_coef, L.c.xZ.xR, L_coef):
            return False
        if not self._equal_mat(U.c.xZ.xZ, U_coef, L.c.xZ.xZ, L_coef):
            return False
        if not self._equal_mat(U.c.xZ.xB, U_coef, L.c.xZ.xB, L_coef):
            return False
        if not self._equal_mat(U.c.xB.xR, U_coef, L.c.xB.xR, L_coef):
            return False
        if not self._equal_mat(U.c.xB.xZ, U_coef, L.c.xB.xZ, L_coef):
            return False
        if not self._equal_mat(U.c.xB.xB, U_coef, L.c.xB.xB, L_coef):
            return False
        return True

    def _equal_mat(self, U, U_coef, L, L_coef):
        if U is None and L is None:
            return True
        if U is None or L is None:
            return False

        Ucoo = U.tocoo()
        Umap = {}
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
            Umap[i,j] = v
        Lcoo = L.tocoo()
        Lmap = {}
        for i,j,v in zip(Acoo.row, Acoo.col, Acoo.data):
            Lmap[i,j] = v

        for i,j in Umap:
            if math.fabs(Umap[i,j]*U_coef + Lmap[i,j]*L_ceof) > 1e-16:
                return False
        return True


if __name__ == "__main__":              # pragma: no cover
    prob = LinearBilevelProblem()
    U = prob.add_upper(3,2,1)
    U.xR.upper_bounds = np.array([1.5, 2.4, 3.1])
    L = prob.add_lower(1,2,3)
    L.xZ.lower_bounds = np.array([1, -2])
    prob.print()
