import numpy as np
import pprint

class LevelVariable(object):

    def __init__(self, num, lb=None, ub=None):
        self.num = num
        self.lower_bounds = lb
        self.upper_bounds = ub

    def print(self, type):
        print("  %s Variables:" % type)
        print("    num: "+str(self.num))
        if self.lower_bounds is not None:
            print("    lower bounds: "+str(self.lower_bounds))
        if self.upper_bounds is not None:
            print("    upper bounds: "+str(self.upper_bounds))

    def __setattr__(self, name, value):
        if name == 'lower_bounds' and value is not None:
            assert (value.size == self.num), "The variable has length %s but specifying a lower bounds with length %s" % (str(self.num), str(value.size))
            super().__setattr__(name, value)
        elif name == 'upper_bounds' and value is not None:
            assert (value.size == self.num), "The variable has length %s but specifying a upper bounds with length %s" % (str(self.num), str(value.size))
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class LevelValues(object):

    def __init__(self):
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

    def _print_value(self, value, name):
        if value is not None:
            if type(value) is np.ndarray:
                print("    %s:" % name, value)
            else:
                print("    %s:" % name)
                for row in str(value).split('\n'):
                    print("      "+row)


class LevelValueWrapper(object):

    def __init__(self, prefix):
        setattr(self, '_values', {})
        setattr(self, '_prefix', prefix)

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name):
        if name.startswith('_'):
            return getattr(self, name)
        else:
            _values = getattr(self, '_values')
            if name in _values:
                return _values[name]
            _values[name] = LevelValues()
            return _values[name]

    def print(self, *args):
        _values = getattr(self, '_values')
        for name in args:
            if name in _values:
                print("  "+self._prefix+"."+name+":")
                _values[name].print()
            

class LevelRepn(object):

    def __init__(self, nxR, nxZ, nxB):
        self.xR = LevelVariable(nxR)    # continuous variables at this level
        self.xZ = LevelVariable(nxZ)    # integer variables at this level
        self.xB = LevelVariable(nxB)    # binary variables at this level
        self.obj_sense = True           # sense of the objective at this level
        self.c = LevelValueWrapper("c") # linear coefficients at this level
        self.A = LevelValueWrapper("A") # constraint matrices at this level
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
        if self.obj_sense:
            print("  Minimize:")
        else:
            print("  Maximize:")
        self.c.print(*args)

        if len(self.A) > 0:
            print("\nConstraints:")
            self.A.print(*args)
            if self.inequalities:
                print("  <=")
            else:
                print("  ==")
            print("   ",self.b)


class BilevelProblem(object):

    def __init__(self, name=None):
        self.name = name
        self.model = None

    def add_upper(self, nxR=0, nxZ=0, nxB=0):
        self.U = LevelRepn(nxR, nxZ, nxB)
        return self.U

    def add_lower(self, nxR=0, nxZ=0, nxB=0):
        self.L = LevelRepn(nxR, nxZ, nxB)
        return self.L

    def print(self):
        if self.name:
            print("# BilevelProblem: "+name)
        else:
            print("# BilevelProblem: unknown")
        print("")
        print("## Upper Level")
        print("")
        self.U.print("U","L")
        print("")
        print("## Lower Level")
        print("")
        self.L.print("U","L")

    def check(self):
        # Perform sanity checks
        pass

if __name__ == "__main__":
    prob = BilevelProblem()
    U = prob.add_upper(3,2,1)
    U.xR.upper_bounds = np.array([1.5, 2.4, 3.1])
    L = prob.add_lower(1,2,3)
    L.xZ.lower_bounds = np.array([1, -2])
    prob.print()

