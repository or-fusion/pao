import pprint
from scipy.sparse import coo_matrix
import numpy as np


class LevelVariable(object):


    def __init__(self, num, lb=None, ub=None):
        self.num = num
        self.values = [None]*num
        self.lower_bounds = lb
        self.upper_bounds = ub

    def __len__(self):
        return self.num

    def print(self, type):
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

    def print(self):
        self._print_value(self.xR, 'xR')
        self._print_value(self.xB, 'xB')
        self._print_value(self.xZ, 'xZ')

    def __len__(self):
        n = 0
        if self._matrix:
            if self.xR is not None:
                n += self.xR.shape[0]
            if self.xZ is not None:
                n += self.xZ.shape[0]
            if self.xB is not None:
                n += self.xB.shape[0]
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

    def _print_value(self, value, name):
        if value is not None:
            if self._matrix:
                print("    %s:" % name)
                for row in str(value).split('\n'):
                    print("      "+row)
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
            return getattr(self, name)
        else:
            _values = getattr(self, '_values')
            if name in _values:
                return _values[name]
            _values[name] = LevelValues(self._matrix)
            return _values[name]

    def print(self, *args):
        _values = getattr(self, '_values')
        for name in args:
            v = _values.get(name,None)
            if v is not None and len(v) > 0:
                print("  "+self._prefix+"."+name+":")
                v.print()
            

class LinearLevelRepn(object):

    def __init__(self, nxR, nxZ, nxB):
        self.xR = LevelVariable(nxR)    # continuous variables at this level
        self.xZ = LevelVariable(nxZ)    # integer variables at this level
        self.xB = LevelVariable(nxB)    # binary variables at this level
        self.minimize = True            # sense of the objective at this level
        self.c = LevelValueWrapper("c") # linear coefficients at this level
        self.A = LevelValueWrapper("A",
                        matrix=True)    # constraint matrices at this level
        self.b = None                   # RHS of the constraints
        self.inequalities = True        # If True, the constraints are inequalities

    def print(self, *args):
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
        self.c.print(*args)

        if self.b is not None and self.b.size > 0:
            print("\nConstraints: ")
            self.A.print(*args)
            if self.inequalities:
                print("  <=")
            else:
                print("  ==")
            print("   ",self.b)

    def __setattr__(self, name, value):
        if name == 'b' and value is not None:
            value = np.array(value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

class QuadraticLevelRepn(object):

    def __init__(self, nxR, nxZ, nxB):
        super().__init__(nxR, nxZ, nxB)
        self.B = LevelValueWrapper("B") # Quadratic term in objective

    def print(self, *args):
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
        self.c.print(*args)
        self.B.print(*args)

        if len(self.A) > 0:
            print("\nConstraints:")
            self.A.print(*args)
            if self.inequalities:
                print("  <=")
            else:
                print("  ==")
            print("   ",self.b)


class LinearBilevelProblem(object):
    """
    Let
        x   = [U.xR, U.xZ, U.xB, L.xR, L.xZ, L.xB]'                         # dense column vector
        U.c = [U.c.U.xR, U.c.U.xZ, U.c.U.xB, U.c.L.xR, U.c.L.xZ, U.c.L.xB]  # dense row vector
        L.c = [L.c.U.xR, L.c.U.xZ, L.c.U.xB, L.c.L.xR, L.c.L.xZ, L.c.L.xB]  # dense row vector
        U.A = [U.A.U.xR, U.A.U.xZ, U.A.U.xB, U.A.L.xR, U.A.L.xZ, U.A.L.xB]  # sparse matrix
        L.A = [L.A.U.xR, L.A.U.xZ, L.A.U.xB, L.A.L.xR, L.A.L.xZ, L.A.L.xB]  # sparse matrix

    min_{U.x}   U.c * x
    s.t.        U.A * x <= U.b                      # Or ==

                where L.x satisifies

                    min_{L.x}   L.c * x
                    s.t.        L.A * x <= L.b      # Or ==
    """

    def __init__(self, name=None):
        self.name = name
        self.model = None

    def add_upper(self, nxR=0, nxZ=0, nxB=0):
        self.U = LinearLevelRepn(nxR, nxZ, nxB)
        return self.U

    def add_lower(self, nxR=0, nxZ=0, nxB=0):
        self.L = LinearLevelRepn(nxR, nxZ, nxB)
        return self.L

    def print(self):
        if self.name:
            print("# LinearBilevelProblem: "+name)
        else:
            print("# LinearBilevelProblem: unknown")
        print("")
        print("## Upper Level")
        print("")
        self.U.print("U","L")
        print("")
        print("## Lower Level")
        print("")
        self.L.print("U","L")

    def check(self):
        U = self.U
        L = self.L
        #
        # Coefficients for upper-level objective
        #
        assert ((U.c.U.xR is None) or (U.c.U.xR.size == len(U.xR)) or (U.c.U.xR.size == 0)), "Incompatible specification of upper-level coefficients for U.xR"
        assert ((U.c.U.xZ is None) or (U.c.U.xZ.size == len(U.xZ)) or (U.c.U.xZ.size == 0)), "Incompatible specification of upper-level coefficients for U.xZ"
        assert ((U.c.U.xB is None) or (U.c.U.xB.size == len(U.xB)) or (U.c.U.xB.size == 0)), "Incompatible specification of upper-level coefficients for U.xB"
        assert ((U.c.L.xR is None) or (U.c.L.xR.size == len(L.xR)) or (U.c.L.xR.size == 0)), "Incompatible specification of upper-level coefficients for L.xR"
        assert ((U.c.L.xZ is None) or (U.c.L.xZ.size == len(L.xZ)) or (U.c.L.xZ.size == 0)), "Incompatible specification of upper-level coefficients for L.xZ"
        assert ((U.c.L.xB is None) or (U.c.L.xB.size == len(L.xB)) or (U.c.L.xB.size == 0)), "Incompatible specification of upper-level coefficients for L.xB"
        
        # Coefficients for lower-level objective
        #
        assert ((L.c.U.xR is None) or (L.c.U.xR.size == len(U.xR)) or (L.c.U.xR.size == 0)), "Incompatible specification of lower-level coefficients for U.xR"
        assert ((L.c.U.xZ is None) or (L.c.U.xZ.size == len(U.xZ)) or (L.c.U.xZ.size == 0)), "Incompatible specification of lower-level coefficients for U.xZ"
        assert ((L.c.U.xB is None) or (L.c.U.xB.size == len(U.xB)) or (L.c.U.xB.size == 0)), "Incompatible specification of lower-level coefficients for U.xB"
        assert ((L.c.L.xR is None) or (L.c.L.xR.size == len(L.xR)) or (L.c.L.xR.size == 0)), "Incompatible specification of lower-level coefficients for L.xR"
        assert ((L.c.L.xZ is None) or (L.c.L.xZ.size == len(L.xZ)) or (L.c.L.xZ.size == 0)), "Incompatible specification of lower-level coefficients for L.xZ"
        assert ((L.c.L.xB is None) or (L.c.L.xB.size == len(L.xB)) or (L.c.L.xB.size == 0)), "Incompatible specification of lower-level coefficients for L.xB"
        #
        # Ncols of upper-level constraints
        #
        assert ((U.A.U.xR is None) or (U.A.U.xR.shape[1] == len(U.xR)) or (U.c.U.xR.shape[1] == 0)), "Incompatible specification of U.A.U.xR and U.xR"
        assert ((U.A.U.xZ is None) or (U.A.U.xZ.shape[1] == len(U.xZ)) or (U.c.U.xZ.shape[1] == 0)), "Incompatible specification of U.A.U.xZ and U.xZ"
        assert ((U.A.U.xB is None) or (U.A.U.xB.shape[1] == len(U.xB)) or (U.c.U.xB.shape[1] == 0)), "Incompatible specification of U.A.U.xB and U.xB"
        assert ((U.A.L.xR is None) or (U.A.L.xR.shape[1] == len(L.xR)) or (U.c.L.xR.shape[1] == 0)), "Incompatible specification of U.A.L.xR and L.xR"
        assert ((U.A.L.xZ is None) or (U.A.L.xZ.shape[1] == len(L.xZ)) or (U.c.L.xZ.shape[1] == 0)), "Incompatible specification of U.A.L.xZ and L.xZ"
        assert ((U.A.L.xB is None) or (U.A.L.xB.shape[1] == len(L.xB)) or (U.c.L.xB.shape[1] == 0)), "Incompatible specification of U.A.L.xB and L.xB"
        #
        # Ncols of lower-level constraints
        #
        assert ((U.A.U.xR is None) or (L.A.U.xR.shape[1] == len(U.xR)) or (L.c.U.xR.shape[1] == 0)), "Incompatible specification of L.A.U.xR and U.xR"
        assert ((U.A.U.xZ is None) or (L.A.U.xZ.shape[1] == len(U.xZ)) or (L.c.U.xZ.shape[1] == 0)), "Incompatible specification of L.A.U.xZ and U.xZ"
        assert ((U.A.U.xB is None) or (L.A.U.xB.shape[1] == len(U.xB)) or (L.c.U.xB.shape[1] == 0)), "Incompatible specification of L.A.U.xB and U.xB"
        assert ((U.A.L.xR is None) or (L.A.L.xR.shape[1] == len(L.xR)) or (L.c.L.xR.shape[1] == 0)), "Incompatible specification of L.A.L.xR and L.xR"
        assert ((U.A.L.xZ is None) or (L.A.L.xZ.shape[1] == len(L.xZ)) or (L.c.L.xZ.shape[1] == 0)), "Incompatible specification of L.A.L.xZ and L.xZ"
        assert ((U.A.L.xB is None) or (L.A.L.xB.shape[1] == len(L.xB)) or (L.c.L.xB.shape[1] == 0)), "Incompatible specification of L.A.L.xB and L.xB"
        #
        # Nrows of upper-level constraints
        #
        if U.b is None:
            assert (U.A.U.xR is None), "Incompatible specification of U.b and U.A.U.xR"
            assert (U.A.U.xZ is None), "Incompatible specification of U.b and U.A.U.xZ"
            assert (U.A.U.xB is None), "Incompatible specification of U.b and U.A.U.xB"
            assert (U.A.L.xR is None), "Incompatible specification of U.b and U.A.L.xR"
            assert (U.A.L.xZ is None), "Incompatible specification of U.b and U.A.L.xZ"
            assert (U.A.L.xB is None), "Incompatible specification of U.b and U.A.L.xB"
        else:
            nr = U.b.size
            if U.A.U.xR is not None:
                assert (nr == U.A.U.xR.shape[0]), "Incompatible specification of U.b and U.A.U.xR"
            if U.A.U.xZ is not None:
                assert (nr == U.A.U.xZ.shape[0]), "Incompatible specification of U.b and U.A.U.xZ"
            if U.A.U.xB is not None:
                assert (nr == U.A.U.xB.shape[0]), "Incompatible specification of U.b and U.A.U.xB"
            if U.A.L.xR is not None:
                assert (nr == U.A.L.xR.shape[0]), "Incompatible specification of U.b and U.A.L.xR"
            if U.A.L.xZ is not None:
                assert (nr == U.A.L.xZ.shape[0]), "Incompatible specification of U.b and U.A.L.xZ"
            if U.A.L.xB is not None:
                assert (nr == U.A.L.xB.shape[0]), "Incompatible specification of U.b and U.A.L.xB"
        #
        # Nrows of lower-level constraints
        #
        if L.b is None:
            assert (L.A.U.xR is None), "Incompatible specification of L.b and L.A.U.xR"
            assert (L.A.U.xZ is None), "Incompatible specification of L.b and L.A.U.xZ"
            assert (L.A.U.xB is None), "Incompatible specification of L.b and L.A.U.xB"
            assert (L.A.L.xR is None), "Incompatible specification of L.b and L.A.L.xR"
            assert (L.A.L.xZ is None), "Incompatible specification of L.b and L.A.L.xZ"
            assert (L.A.L.xB is None), "Incompatible specification of L.b and L.A.L.xB"
        else:
            nr = L.b.size
            if L.A.U.xR is not None:
                assert (nr == L.A.U.xR.shape[0]), "Incompatible specification of L.b and L.A.U.xR"
            if L.A.U.xZ is not None:
                assert (nr == L.A.U.xZ.shape[0]), "Incompatible specification of L.b and L.A.U.xZ"
            if L.A.U.xB is not None:
                assert (nr == L.A.U.xB.shape[0]), "Incompatible specification of L.b and L.A.U.xB"
            if L.A.L.xR is not None:
                assert (nr == L.A.L.xR.shape[0]), "Incompatible specification of L.b and L.A.L.xR"
            if L.A.L.xZ is not None:
                assert (nr == L.A.L.xZ.shape[0]), "Incompatible specification of L.b and L.A.L.xZ"
            if L.A.L.xB is not None:
                assert (nr == L.A.L.xB.shape[0]), "Incompatible specification of L.b and L.A.L.xB"

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
            if math.fabs(U[i]*U_coef + L[i]*L_ceof) > 1e-16:
                return False
        return True


class QuadraticBilevelProblem(LinearBilevelProblem):
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


if __name__ == "__main__":
    prob = LinearBilevelProblem()
    U = prob.add_upper(3,2,1)
    U.xR.upper_bounds = np.array([1.5, 2.4, 3.1])
    L = prob.add_lower(1,2,3)
    L.xZ.lower_bounds = np.array([1, -2])
    prob.print()

