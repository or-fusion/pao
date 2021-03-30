import math
import weakref
import copy
import pprint
import collections.abc
from scipy.sparse import csr_matrix, dok_matrix
import numpy as np
from pyutilib.misc import Bunch


def _equal_nparray(Ux, U_coef, Lx, L_coef):
    if Ux is None and Lx is None:
        return True
    if Ux is None or Lx is None:
        return False
    for i in range(Ux.size):
        if math.fabs(Ux[i]*U_coef + Lx[i]*L_coef) > 1e-16:
            return False
    return True

def _update_matrix(*, A, old, new, update_columns=True):
    A = A.todok()
    if not update_columns:
        A = A.transpose()

    A_ = dok_matrix((A.shape[0], new.nxR+new.nxZ+new.nxB), dtype=np.float64)
    for ndx in A.keys():
        i,j = ndx
        if j < old.nxR:
            if j<new.nxR:
                A_[i,j] = A[i,j]
        elif j < old.nxR+old.nxZ:
            j_ = j-old.nxR
            if j_<new.nxZ:
                A_[i,j_+new.nxR] = A[i,j]
        else:
            j_ = j-old.nxR-old.nxZ
            if j_<new.nxB:
                A_[i,j_+new.nxR+new.nxZ] = A[i,j]

    if not update_columns:
        A_ = A_.transpose()
    return A_.tocsr()


class SimplifiedList(collections.abc.MutableSequence):
    """
    This is a normal list class, except if the user asks to
    get attributes, then they will be collected from the
    first list element.  However, in that case, an error
    is generated if there are more than one list elements.
    """

    def __init__(self):
        self._data = []

    def clone(self, parent=None, clone_fn=None):
        ans = SimplifiedList()
        ans._data = [ val.clone(parent=parent, clone_fn=clone_fn) for val in self._data ]
        return ans

    def append(self, val):
        self._data.append(val)

    def insert(self, i, val):
        self._data.insert(i, val)

    def __iter__(self):
        for v in self._data:
            yield v

    def __getitem__(self, i):
        if i >= len(self._data):
            raise IndexError
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
        self.pyvar = [None]*self.num
        if lb is None:
            self.lower_bounds = np.array([np.NINF]*self.num, dtype=np.float64)
        else:
            self.lower_bounds = lb
        if ub is None:
            self.upper_bounds = np.array([np.PINF]*self.num, dtype=np.float64)
        else:
            self.upper_bounds = ub
        for i in range(nxB):
            self.lower_bounds[i+nxR+nxZ] = 0
            self.upper_bounds[i+nxR+nxZ] = 1

    def clone(self):
        ans = LevelVariable(self.nxR, self.nxZ, self.nxB, np.copy(self.lower_bounds), np.copy(self.upper_bounds))
        ans.values = copy.copy(self.values)
        ans.pyvar = self.pyvar
        return ans

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield i

    def _resize(self, nxR, nxZ, nxB, lb=np.NINF, ub=np.PINF):
        if nxR == self.nxR and nxZ == self.nxR and nxB == self.nxB:     # pragma: nocover
            return

        num = nxR+nxZ+nxB
        curr_num = self.num
        self.num = num
        self.values = [None]*num
        self.pyvar = [None]*num

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

    def print(self):                  # pragma: no cover
        print("  nxR: "+str(self.nxR))
        print("  nxZ: "+str(self.nxZ))
        print("  nxB: "+str(self.nxB))
        print("  lower bounds: "+str(self.lower_bounds))
        print("  upper bounds: "+str(self.upper_bounds))
        print("  nonzero values:")
        for i,v in enumerate(self.values):
            if v is not None and v != 0:
                if type(v) is int:
                    print("    %d: %d" % (i, v))
                else:
                    print("    %d: %f" % (i, v))

    def __setattr__(self, name, value):
        if name == 'lower_bounds':
            assert (value is not None), "Cannot specify null lower bounds array"
            # Add this check in the model checks
            assert (len(value) == self.num), "The variable has length %s but specifying a lower bounds with length %s" % (str(self.num), str(len(value)))
            if type(value) is list:
                value = np.array(value, dtype=np.float64)
            super().__setattr__(name, value)
        elif name == 'upper_bounds':
            assert (value is not None), "Cannot specify null upper bounds array"
            # Add this check in the model checks
            assert (len(value) == self.num), "The variable has length %s but specifying a upper bounds with length %s" % (str(self.num), str(len(value)))
            if type(value) is list:
                value = np.array(value, dtype=np.float64)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class LevelValues(object):

    def __init__(self, matrix_list=False, matrix=False, x=None):
        self._matrix = matrix
        self._matrix_list = matrix_list
        self.set_values(x)

    def clone(self):
        ans = LevelValues(matrix_list=self._matrix_list, matrix=self._matrix)
        if self._matrix:
            if self.x is not None:
                ans.x = self.x.copy()
        elif self._matrix_list:
            ans.x = [None if m is None else np.copy(m) for m in self.x]
        else:
            if self.x is not None:
                ans.x = np.copy(self.x)
        return ans

    def set_values(self, x=None):
        if type(x) is list:
            if self._matrix:                
                x = csr_matrix( x )
            elif self._matrix_list:                
                x = [csr_matrix(m) if type(m) is list else m for m in x]
            else:
                x = np.array(x, dtype=np.float64)
        elif type(x) is tuple:
            if self._matrix:
                nrows,ncols = x[0]
                xnew = dok_matrix((nrows,ncols))
                for key,value in x[1].items():
                    xnew[key] = value
                x = xnew
            elif self._matrix_list:
                ncon,nrows,ncols = x[0]
                con = {}
                for key,value in x[1].items():
                    i,j,k = key
                    if i not in con:
                        con[i] = dok_matrix((nrows,ncols))
                    con[i][j,k] = value
                xnew = []
                for i in range(ncon):
                    if i in con:
                        xnew.append( csr_matrix(con[i]) )
                    else:
                        xnew.append( None )
                x = xnew
                
        super().__setattr__('x', x)

    def __setattr__(self, name, value):
        if name == 'x':
            self.set_values(value)
        else:
            super().__setattr__(name, value)

    def print(self, prefix):                        # pragma: no cover
        self._print_value(self.x, prefix)

    def __len__(self):
        n = 0
        if self._matrix:
            if self.x is not None:
                n = max(n, self.x.shape[0])
        elif self._matrix_list:
            if self.x is not None:
                n += sum(0 if m is None else m.size for m in self.x)
        else:
            if self.x is not None:
                n += self.x.size
        return n

    def _print_value(self, value, name):            # pragma: no cover
        if value is not None:
            if self._matrix:
                print("    %s:" % name)
                if value.size:
                    print("        shape: %d %d" % (value.shape[0], value.shape[1]))
                print("        nonzeros:")
                for row in value.toarray().tolist():
                    print("        ",row)
            elif self._matrix_list:
                print("    %s:" % name)
                for i,m in enumerate(value):
                    print("    %d:" % i)
                    if m is None:
                        print("        None")
                        continue
                    print("        shape: %d %d" % (m.shape[0], m.shape[1]))
                    print("        nonzeros:")
                    for row in str(m).split('\n'):
                        print("        "+row)
            else:
                print("    %s:" % name, value)


class LevelValueWrapper1(object):

    def __init__(self, prefix, matrix=False):
        setattr(self, '_matrix', matrix)
        setattr(self, '_values', {})
        setattr(self, '_prefix', prefix)

    def __len__(self):
        _values = getattr(self, '_values')
        return len(_values)

    def __iter__(self):
        yield from self._values.keys()

    def __getattr__(self, name):
        if name.startswith('_'):                # pragma: no cover
            return super().__getattr__(name)
        else:
            raise AttributeError("No attributes in this object")

    def __getitem__(self, lvl):
        if type(lvl) is int:
            i = lvl
        else:
            i = lvl.id
        _values = self._values
        retval = _values.get(i, None)
        if retval is None:
            return None
        return retval.x

    def __setitem__(self, lvl, value):
        if type(lvl) is int:
            i = lvl
        else:
            i = lvl.id
        _values = self._values
        if value is None:
            if i in _values:
                del _values[i]
        else:
            _values[i] = LevelValues(matrix=self._matrix, x=value)

    def clone(self):
        ans = LevelValueWrapper1(self._prefix, matrix=self._matrix)
        for name in self._values:
            ans._values[name] = self._values[name].clone()
        return ans

    def print(self, names):               # pragma: no cover
        _values = self._values
        first = True
        for i,name in names:
            v = _values.get(i,None)
            if v is None:
                continue
            if len(v) > 0:
                if first:
                    print("  "+self._prefix+":")
                    first = False
                v.print(name)


class LevelValueWrapper2(object):

    def __init__(self, prefix, matrix=True):
        setattr(self, '_matrix', matrix)
        setattr(self, '_values', {})
        setattr(self, '_prefix', prefix)

    def __len__(self):
        _values = getattr(self, '_values')
        return len(_values)

    def __iter__(self):
        yield from self._values.keys()

    def __getattr__(self, name):
        if name.startswith('_'):                # pragma: no cover
            return super().__getattr__(name)
        else:
            raise AttributeError("No attributes in this object")

    def __getitem__(self, lvls):
        lvl1, lvl2 = lvls
        if type(lvl1) is int:
            i = lvl1
        else:
            i = lvl1.id
        if type(lvl2) is int:
            j = lvl2
        else:
            j = lvl2.id
        _values = self._values
        retval = _values.get((i,j), None)
        if retval is None:
            return None
        return retval.x

    def __setitem__(self, lvls, value):
        lvl1, lvl2 = lvls
        if type(lvl1) is int:
            i = lvl1
        else:
            i = lvl1.id
        if type(lvl2) is int:
            j = lvl2
        else:
            j = lvl2.id
        assert (i<=j), "Require quadratic terms to be indexed with upper-level variables first"
        if not type(lvl1) is int and not type(lvl2) is int:
            assert (lvl1.id in [L.id for L in lvl2.levels()]), "Require quadratic terms to be in the same subproblem tree"
        if value is None:
            if (i,j) in self._values:
                del self._values[i,j]
        else:
            self._values[i,j] = LevelValues(matrix=self._matrix, matrix_list=not self._matrix, x=value)

    def clone(self):
        ans = LevelValueWrapper2(self._prefix, matrix=self._matrix)
        for name in self._values:
            ans._values[name] = self._values[name].clone()
        return ans

    def print(self, names):               # pragma: no cover
        _values = self._values
        first = True
        for i,name in names:
            v = _values.get(i,None)
            if v is None:
                continue
            if len(v) > 0:
                if first:
                    print("  "+self._prefix+":")
                    first = False
                v.print(name)


class LinearLevelRepn(object):

    _counter = 0

    def __init__(self, nxR, nxZ, nxB, id=None):
        if id is None:
            self.id = LinearLevelRepn._counter
            LinearLevelRepn._counter += 1
        else:
            self.id = id

        self.x = LevelVariable(nxR, nxZ, nxB)    # variables at this level
        self.c = LevelValueWrapper1("c") # objective coefficients at this level
        self.A = LevelValueWrapper1("A",
                        matrix=True)    # constraint matrices at this level
        self.b = np.ndarray(0, dtype=np.float64)          # RHS of the constraints
        self._minimize = True           # sense of the objective at this level
        self._inequalities = True       # If True, the constraints are inequalities
        self.d = 0                      # constant in objective at this level

        self.LL = SimplifiedList()      # lower levels
        self.UL = lambda: None          # "empty weakref" to upper level

        self.name = None                # a string descriptor for this level

    def _add_lower(self, tmp, nxR=0, nxZ=0, nxB=0, name=None, id=None):
        if name is None:
            tmp.name = self.name + ".LL[%d]" % len(self.LL)
        else:
            tmp.name = name
        tmp.UL = weakref.ref(self)
        self.LL.append(tmp)
        return tmp

    def add_lower(self, *, nxR=0, nxZ=0, nxB=0, name=None, id=None):
        return self._add_lower(LinearLevelRepn(nxR, nxZ, nxB, id=id), nxR=nxR, nxZ=nxZ, name=name, id=id)

    @property
    def inequalities(self):
        return self._inequalities

    @inequalities.setter
    def inequalities(self, val):
        self._inequalities = val

    @property
    def equalities(self):
        return not self._inequalities

    @equalities.setter
    def equalities(self, val):
        self._inequalities = not val

    @property
    def minimize(self):
        return self._minimize

    @minimize.setter
    def minimize(self, val):
        self._minimize = val

    @property
    def maximize(self):
        return not self._minimize

    @maximize.setter
    def maximize(self, val):
        self._minimize = not val

    def resize(self, *, nxR=0, nxZ=0, nxB=0, lb=np.NINF, ub=np.PINF):
        old = Bunch(nxR=self.x.nxR, nxZ=self.x.nxZ, nxB=self.x.nxB)
        new = Bunch(nxR=nxR, nxZ=nxZ, nxB=nxB)
        self.x._resize(nxR=nxR, nxZ=nxZ, nxB=nxB, lb=lb, ub=ub)
        #
        # Walk the tree, updating c[self] and A[self]
        #
        for L in self.levels():
            L._update(level=self, new=new, old=old)

    def _update(self, *, level, new, old):
        #
        # Update 'c'
        #
        c = self.c[level]
        if c is not None:
            c_ = np.zeros(new.nxR+new.nxZ+new.nxB)          # RHS of the constraints
            for i in range(min(new.nxR,old.nxR)):
                c_[i] = c[i]
            for i in range(min(new.nxZ,old.nxZ)):
                c_[i+new.nxR] = c[i+old.nxR]
            for i in range(min(new.nxB,old.nxB)):
                c_[i+new.nxR+new.nxZ] = c[i+old.nxR+old.nxZ]
            self.c[level] = c_
        #
        # Update 'A'
        #
        A = self.A[level]
        if A is not None:
            self.A[level] = _update_matrix(A=A, old=old, new=new)

    #
    # Iterate over the sublevels and parents in DFS order
    #
    def levels(self, parents=True):
        for X in self._sublevels():
            yield X
        if parents:
            X = self.UL()
            while X is not None:
                yield X
                X = X.UL()

    #
    # Iterate over sublevels in DFS order
    #
    def _sublevels(self):
        yield self
        for L in self.LL:
            for X in L._sublevels():
                yield X

    @staticmethod
    def _clone_level(self, parent=None, data=[], ans=None):
        if ans is None:
            ans = LinearLevelRepn(0,0,0)
        ans.x = self.x.clone()
        ans.c = self.c.clone()
        ans.A = self.A.clone()
        ans.b = np.copy(self.b)
        ans.minimize = self.minimize
        ans.inequalities = self.inequalities
        ans.d = self.d
        if parent is None:
            ans.UL = lambda: None # "empty weakref"
        else:
            ans.UL = weakref.ref(parent)
        # TODO - Should we allow users to annotate these objects with other data?
        for attr in dir(self):
            if attr in data:
                continue
            if attr in ['clone', 'print', 'resize', 'add_lower', 'levels']:    # methods
                continue
            if attr.startswith('_'):
                continue
            setattr(ans, attr, copy.copy(getattr(self, attr)))
        return ans

    def clone(self, parent=None, clone_fn=None):
        if clone_fn is None:
            clone_fn = LinearLevelRepn._clone_level
        ans = clone_fn(self, parent=parent, data=['x', 'c', 'A', 'b', 'minimize', 'inequalities', 'equalities', 'd', 'LL', 'UL'])
        ans.LL = self.LL.clone(parent=ans, clone_fn=clone_fn)
        return ans

    def print(self, names):       # pragma: no cover
        print("")
        print("## Level: "+self.name)
        print("")

        print("Variables:")
        self.x.print()

        print("\nObjective:")
        if self.minimize:
            print("  Minimize:")
        else:
            print("  Maximize:")
        self.c.print(names)
        print("  d:",self.d)

        if self.b.size > 0:
            print("\nConstraints: ")
            self.A.print(names)
            if self.inequalities:
                print("  <=", self.b)
            else:
                print("  ==", self.b)
        #
        # Recurse
        #
        for L in self.LL:
            L.print(names)

    def __setattr__(self, name, value):
        if name == 'b' and value is not None:
            value = np.array(value, dtype=np.float64)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def check(self):
        #
        # Size of 'c'
        #
        for X in self.levels():
            assert ((self.c[X] is None) or (self.c[X].size == len(X.x)) or (self.c[X].size == 0)), "Incompatible specification of coefficients for %s.c[%s]: %d != %d" % (self.name, X.name, self.c[X].size, len(X.x))
        #
        # Ncols of 'A'
        #
        for X in self.levels():
            assert ((self.A[X] is None) or (self.A[X].shape[1] == len(X.x))), "Incompatible specification of %s.A[%s] and %s.x (%d != %d)" % (self.name, X.name, X.name, self.A[X].shape[1], len(X.x))
        #
        # Nrows of 'A'
        #
        if self.b is None:
            for X in self.levels():
                assert (self.A[X] is None), "Incompatible specification of %s.b and %s.A[%s]" % (self.name, self.name, X.name)
        else:
            nr = self.b.size
            for X in self.levels():
                if self.A[X] is not None:
                    assert (nr == self.A[X].shape[0]), "Incompatible specification of %s.b and %s.A[%s] (%d != %d)" % (self.name, self.name, X.name, nr, self.A[X].shape[0])


class QuadraticLevelRepn(LinearLevelRepn):

    def __init__(self, nxR, nxZ, nxB, id=None):
        LinearLevelRepn.__init__(self, nxR, nxZ, nxB, id)
        self.P = LevelValueWrapper2("P", matrix=True)        # objective quadratic terms
        self.Q = LevelValueWrapper2("Q", matrix=False)       # constraint quadratic terms

    def add_lower(self, *, nxR=0, nxZ=0, nxB=0, name=None, id=None):
        return self._add_lower(QuadraticLevelRepn(nxR, nxZ, nxB, id=id), nxR=nxR, nxZ=nxZ, name=name, id=id)

    def _update(self, *, level, new, old):
        #
        # Update 'c' and 'A'
        #
        LinearLevelRepn._update(self, level=level, new=new, old=old)
        #
        # Update 'P'
        #
        for L1,L2 in self.P:
            if L1 == level.id or L2 == level.id:
                self.P[L1,L2] = _update_matrix(A=self.P[L1,L2], old=old, new=new, update_columns=L2==level.id)
        #
        # Update 'Q'
        #
        for L1,L2 in self.Q:
            if L1 == level.id or L2 == level.id:
                Q = self.Q[L1,L2]
                self.Q[L1,L2] = [None if m is None else _update_matrix(A=m, old=old, new=new, update_columns=L2==level.id) for m in Q]

    @staticmethod
    def _clone_level(self, parent, data, ans=None):
        if ans is None:
            ans = QuadraticLevelRepn(0,0,0)
        LinearLevelRepn._clone_level(self, parent, data, ans=ans)
        ans.P = self.P.clone()
        ans.Q = self.Q.clone()
        return ans
        
    def clone(self, parent=None, clone_fn=None):
        if clone_fn is None:
            clone_fn = QuadraticLevelRepn._clone_level
        ans = clone_fn(self, parent=parent, data=['x', 'c', 'A', 'b', 'minimize', 'maximize', 'inequalities', 'equalities', 'd', 'LL', 'UL', 'P', 'Q'])
        ans.LL = self.LL.clone(parent=ans, clone_fn=clone_fn)
        return ans

    def print(self, names):       # pragma: no cover
        print("")
        print("## Level: "+self.name)
        print("")

        print("Variables:")
        self.x.print()

        print("\nObjective:")
        if self.minimize:
            print("  Minimize:")
        else:
            print("  Maximize:")
        self.c.print(names)
        self.P.print(names)
        print("  d:",self.d)

        if self.b.size > 0:
            print("\nConstraints: ")
            self.A.print(names)
            self.Q.print(names)
            if self.inequalities:
                print("  <=", self.b)
            else:
                print("  ==", self.b)
        #
        # Recurse
        #
        for L in self.LL:
            L.print(names)

    def check(self):
        LinearLevelRepn.check(self)
        #
        # Ncols of 'P'
        #
        for X in self.levels():
            if X.id <= self.id:
                assert ((self.P[X,self] is None) or (self.P[X,self].shape[1] == len(self.x))), "Incompatible specification of columns for %s.P[%s,%s] and %s.x (%d != %d)" % (self.name, X.name, self.name, self.name, self.P[X,self].shape[1], len(self.x))


class LinearMultilevelProblem(object):
    """
    ::

      For bilevel problems, let:

        U   = LinearMultilevelProblem.U
        L   = U.L
        x   = [U.x, L.x]'       # dense column vector
        U.c = [U.c[U], U.c[L]]' # dense column vector
        L.c = [L.c[U], L.c[L]]' # dense column vector
        U.A = [U.A[U], U.A[L]]  # sparse matrix
        L.A = [L.A[U], L.A[L]]  # sparse matrix

      Then we have:

        min_{U.x}   U.c' * x + U.d
        s.t.        U.A  * x       <= U.b                  # Or ==

                where L.x satisifies

                    min_{L.x}   L.c' * x + L.d
                    s.t.        L.A  * x       <= L.b      # Or ==
    """

    def __init__(self, name=None):
        self.name = name
        self.U = None

    def add_upper(self, *, nxR=0, nxZ=0, nxB=0, name=None, id=None):
        assert (self.U is None), "Cannot create a second upper-level in a LinearMultilevelProblem"
        self.U = LinearLevelRepn(nxR, nxZ, nxB, id=id)
        if name is None:
            self.U.name = "U"
        else:
            self.U.name = name
        return self.U

    def levels(self):
        yield from self.U._sublevels()

    def clone(self, clone_fn=None):
        ans = LinearMultilevelProblem()
        ans.name = self.name
        ans.U = self.U.clone(clone_fn=clone_fn)
        return ans

    def print(self):                            # pragma: no cover
        nL = len(self.U.LL)
        if self.name:
            print("# LinearMultilevelProblem: "+self.name)
        else:
            print("# LinearMultilevelProblem: unknown")

        names = [(L.id,L.name) for L in self.levels()]
        self.U.print(names)

    def check(self):                    # pragma: no cover
        for L in self.levels():
            L.check()

    def check_opposite_objectives(self, U, L):
        if id(U.c) == id(L.c) and L.minimize ^ U.minimize:
            return True
        U_coef = 1 if U.minimize else -1
        L_coef = 1 if L.minimize else -1
        if not _equal_nparray(U.c[U], U_coef, L.c[U], L_coef):
            return False
        if not _equal_nparray(U.c[L], U_coef, L.c[L], L_coef):
            return False
        return True


class QuadraticMultilevelProblem(object):
    """
    ::
 
      For bilevel problems, let:
    
        U   = QuadraticMultilevelProblem.U
        L   = U.L
        x   = [U.x, L.x]'           # dense column vector
        U.c = [U.c[U], U.c[L]]'     # dense column vector
        U.P = [U.P[U,U], U.P[U,L]]  # sparse matrix
              [0,        U.P[L,L]]
        U.A = [U.A[U], U.A[L]]      # sparse matrix
        U.Q = [U.Q[U,U], U.Q[U,L]]  # sparse matrix
              [0,        U.Q[L,L]]
        L.c = [L.c[U], L.c[L]]'     # dense column vector
        L.P = [L.P[U,U], L.P[U,L]]  # sparse matrix
              [0,        L.P[L,L]]
        L.A = [L.A[U], L.A[L]]      # sparse matrix
        L.Q = [L.Q[U,U], L.Q[U,L]]  # sparse matrix
              [0,        L.Q[L,L]]
    
      Then we have:
    
        min_{U.x}   U.c' * x + x' * U.P * x + U.d
        s.t.        U.A  * x + x' * U.Q * x       <= U.b                 # Or ==

                where L.x satisifies

                    min_{L.x}   L.c' * x + x' * L.P * x + L.d
                    s.t.        L.A  * x + x' * L.Q * x       <= L.b     # Or ==
    """

    def __init__(self, name=None, bilinear=False):
        self.bilinear = bilinear
        self.name = name
        self.U = None

    def add_upper(self, *, nxR=0, nxZ=0, nxB=0, name=None, id=None):
        assert (self.U is None), "Cannot create a second upper-level in a QuadraticMultilevelProblem"
        self.U = QuadraticLevelRepn(nxR, nxZ, nxB, id=id)
        if name is None:
            self.U.name = "U"
        else:
            self.U.name = name
        return self.U

    def levels(self):
        yield from self.U._sublevels()

    def clone(self, clone_fn=None):
        ans = QuadraticMultilevelProblem()
        ans.name = self.name
        ans.U = self.U.clone(clone_fn=clone_fn)
        return ans

    def print(self):                            # pragma: no cover
        nL = len(self.U.LL)
        if self.name:
            print("# QuadraticMultilevelProblem: "+self.name)
        else:
            print("# QuadraticMultilevelProblem: unknown")

        names = [(L.id,L.name) for L in self.levels()]
        for L in self.levels():
            for X in L.levels():
                if L.id <= X.id:
                    names.append( ((L.id,X.id), L.name+","+X.name) )
        self.U.print(names)

    def check(self):                    # pragma: no cover
        for L in self.levels():
            L.check()

    def check_opposite_objectives(self, U, L):
        if id(U.c) == id(L.c) and L.minimize ^ U.minimize:
            return True
        U_coef = 1 if U.minimize else -1
        L_coef = 1 if L.minimize else -1
        if not _equal_nparray(U.c[U], U_coef, L.c[U], L_coef):
            return False
        if not _equal_nparray(U.c[L], U_coef, L.c[L], L_coef):
            return False
        return True


if __name__ == "__main__":              # pragma: no cover
    prob = LinearMultilevelProblem()
    U = prob.add_upper(nxR=3,nxZ=2,nxB=1)
    U.x.upper_bounds = np.array([1.5, 2.4, 3.1, np.PINF, np.PINF, 1])
    L = U.add_lower(nxR=1,nxZ=2,nxB=3)
    L.x.lower_bounds = np.array([np.NINF, 1, -2, 0, 0, 0])
    prob.print()

