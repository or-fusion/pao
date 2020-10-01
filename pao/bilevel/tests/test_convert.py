import numpy as np
import pyutilib.th as unittest
import pyomo.environ as pe
from pao.bilevel.convert import collect_multilevel_tree, convert_pyomo2LinearBilevelProblem
from pao.bilevel import SubModel
from pao.lbp import LinearBilevelProblem


class TestMultilevelTree(unittest.TestCase):

    def test1(self):
        M = pe.ConcreteModel()
        M.x = pe.Var()
        M.o = pe.Objective(expr=M.x)
        
        var = {}
        tree = collect_multilevel_tree(M, var)

        self.assertEqual(len(tree.children), 0)
        self.assertEqual(len(var), 1)
        self.assertEqual(len(tree.orepn), 1)
        self.assertEqual(len(tree.crepn), 0)
        self.assertEqual(len(tree.fixedvars), 0)
        self.assertEqual(len(tree.unfixedvars), 1)
        self.assertEqual(len(tree.xR), 1)
        self.assertEqual(len(tree.xZ), 0)
        self.assertEqual(len(tree.xB), 0)

    def test2(self):
        # Only M.x is used in an expression
        M = pe.ConcreteModel()
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.o = pe.Objective(expr=M.x)
        
        var = {}
        tree = collect_multilevel_tree(M, var)

        self.assertEqual(len(tree.children), 0)
        self.assertEqual(len(var), 1)
        self.assertEqual(len(tree.orepn), 1)
        self.assertEqual(len(tree.crepn), 0)
        self.assertEqual(len(tree.fixedvars), 0)
        self.assertEqual(len(tree.unfixedvars), 1)
        self.assertEqual(len(tree.xR), 1)
        self.assertEqual(len(tree.xZ), 0)
        self.assertEqual(len(tree.xB), 0)

    def test3(self):
        # All variables used in an expression
        M = pe.ConcreteModel()
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.o = pe.Objective(expr=M.x+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z))
        
        var = {}
        tree = collect_multilevel_tree(M, var)

        self.assertEqual(len(tree.children), 0)
        self.assertEqual(len(var), 6)
        self.assertEqual(len(tree.orepn), 1)
        self.assertEqual(len(tree.crepn), 0)
        self.assertEqual(len(tree.fixedvars), 0)
        self.assertEqual(len(tree.unfixedvars), 6)
        self.assertEqual(len(tree.xR), 1)
        self.assertEqual(len(tree.xZ), 3)
        self.assertEqual(len(tree.xB), 2)

    def test4(self):
        # All variables used in an expression
        # One is fixed 
        M = pe.ConcreteModel()
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.z[4].fix(1)
        M.o = pe.Objective(expr=M.x+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z))
        
        var = {}
        tree = collect_multilevel_tree(M, var)

        self.assertEqual(len(tree.children), 0)
        self.assertEqual(len(var), 5)
        self.assertEqual(len(tree.orepn), 1)
        self.assertEqual(len(tree.crepn), 0)
        self.assertEqual(len(tree.fixedvars), 0)
        self.assertEqual(len(tree.unfixedvars), 5)
        self.assertEqual(len(tree.xR), 1)
        self.assertEqual(len(tree.xZ), 2)
        self.assertEqual(len(tree.xB), 2)

    def test5(self):
        # All variables used in an expression
        # One of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.z[4].setlb(0)
        M.z[4].setub(1)
        M.o = pe.Objective(expr=M.x+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z))
        
        var = {}
        tree = collect_multilevel_tree(M, var)

        self.assertEqual(len(tree.children), 0)
        self.assertEqual(len(var), 6)
        self.assertEqual(len(tree.orepn), 1)
        self.assertEqual(len(tree.crepn), 0)
        self.assertEqual(len(tree.fixedvars), 0)
        self.assertEqual(len(tree.unfixedvars), 6)
        self.assertEqual(len(tree.xR), 1)
        self.assertEqual(len(tree.xZ), 2)
        self.assertEqual(len(tree.xB), 3)

    def test6a(self):
        # All variables used in an expression
        # One of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.p = pe.Param(initialize=1)
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.z[4].setlb(0)
        M.z[4].setub(1)
        M.o = pe.Objective(expr=M.x+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z))
        M.c1 = pe.Constraint(expr=M.p <= 0)
        
        var = {}
        try:
            tree = collect_multilevel_tree(M, var)
            self.fail("Expected AssertionError")
        except AssertionError:
            pass

    def test6b(self):
        # All variables used in an expression
        # One of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.p = pe.Param(initialize=1)
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.z[4].setlb(0)
        M.z[4].setub(1)
        M.o = pe.Objective(expr=M.x+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z))
        M.c1 = pe.Constraint(expr=pe.inequality(2, M.p, 3))
        
        var = {}
        try:
            tree = collect_multilevel_tree(M, var)
            self.fail("Expected AssertionError")
        except AssertionError:
            pass

    def test6c(self):
        # All variables used in an expression
        # One of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.p = pe.Param(initialize=1)
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.z[4].setlb(0)
        M.z[4].setub(1)
        M.o = pe.Objective(expr=M.x+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z))
        M.c1 = pe.Constraint(expr=M.p == 0)
        
        var = {}
        try:
            tree = collect_multilevel_tree(M, var)
            self.fail("Expected AssertionError")
        except AssertionError:
            pass


    def test_initialize1(self):
        # All variables used in an expression
        # One of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.x = pe.Var()
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z[4].setlb(0)
        M.z[4].setub(1)
        M.o = pe.Objective(expr=2+M.x+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z))
        
        M.s = SubModel(fixed=M.x)

        lbp = convert_pyomo2LinearBilevelProblem(M, True)
        
        self.assertEqual(len(lbp.U.xR), len(M.x))
        self.assertEqual(len(lbp.L.xZ), len(M.z)-1)
        self.assertEqual(len(lbp.L.xB), len(M.y)+1)

        self.assertEqual(lbp.U.d, 2)

        self.assertEqual(list(lbp.U.c.U.xR), [1])
        self.assertEqual(list(lbp.U.c.L.xZ), [3,3])
        self.assertEqual(list(lbp.U.c.L.xB), [2,2,3])

    def test_initialize_2(self):
        # All variables used in an expression
        # None of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.p = pe.Param(initialize=1)
        M.x = pe.Var([0])
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.x[0].setlb(0)
        M.x[0].setub(1)
        M.z[4].setlb(0)
        M.z[4].setub(2)
        M.o = pe.Objective(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z))
        def c_rule(M, i):
            return pe.Constraint.Skip
        M.c1 = pe.Constraint([0], rule=c_rule)
        M.c2 = pe.Constraint(expr=M.x[0] <= np.PINF)
        M.c3 = pe.Constraint(expr=M.p <= 2)
        
        M.s = SubModel(fixed=M.x)

        lbp = convert_pyomo2LinearBilevelProblem(M, True)
        
        self.assertEqual(len(lbp.U.xR), len(M.x))
        self.assertEqual(len(lbp.U.xZ), len(M.z))
        self.assertEqual(len(lbp.U.xB), len(M.y))

        self.assertEqual(lbp.U.d, 0)

        self.assertEqual(list(lbp.U.c.U.xR), [1])
        self.assertEqual(list(lbp.U.c.U.xZ), [3,3,3])
        self.assertEqual(list(lbp.U.c.U.xB), [2,2])

    def test_initialize_2L(self):
        # All variables used in an expression
        # None of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.p = pe.Param(initialize=1)
        M.x = pe.Var([0])
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.x[0].setlb(0)
        M.x[0].setub(1)
        M.z[4].setlb(0)
        M.z[4].setub(2)
        M.o = pe.Objective(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z))
        def c_rule(M, i):
            return pe.Constraint.Skip
        M.c1 = pe.Constraint([0], rule=c_rule)
        M.c2 = pe.Constraint(expr=M.x[0] <= np.PINF)
        M.c3 = pe.Constraint(expr=M.p <= 2)
        
        M.s = SubModel(fixed=M.x)

        M.s.o = pe.Objective(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z))

        M.s.c1 = pe.Constraint([0], rule=c_rule)
        M.s.c2 = pe.Constraint(expr=M.x[0] <= np.PINF)
        M.s.c3 = pe.Constraint(expr=M.p <= 2)

        lbp = convert_pyomo2LinearBilevelProblem(M, True)
        
        self.assertEqual(len(lbp.U.xR), len(M.x))
        self.assertEqual(len(lbp.U.xZ), len(M.z))
        self.assertEqual(len(lbp.U.xB), len(M.y))

        self.assertEqual(lbp.U.d, 0)

        self.assertEqual(list(lbp.U.c.U.xR), [1])
        self.assertEqual(list(lbp.U.c.L.xZ), [3,3,3])
        self.assertEqual(list(lbp.U.c.L.xB), [2,2])

        self.assertEqual(list(lbp.L.c.U.xR), [1])
        self.assertEqual(list(lbp.L.c.L.xZ), [3,3,3])
        self.assertEqual(list(lbp.L.c.L.xB), [2,2])

    def test_initialize_3(self):
        # All variables used in an expression
        # None of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.x = pe.Var([0])
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.x[0].setlb(0)
        M.x[0].setub(1)
        M.z[4].setlb(0)
        M.z[4].setub(2)
        M.o = pe.Objective(expr=1)
        M.c = pe.Constraint(expr=M.x[0]+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z) == 0)
        
        M.s = SubModel(fixed=M.x)

        lbp = convert_pyomo2LinearBilevelProblem(M, True)
        
        self.assertEqual(len(lbp.U.xR), len(M.x))
        self.assertEqual(len(lbp.U.xZ), len(M.z))
        self.assertEqual(len(lbp.U.xB), len(M.y))

        self.assertEqual(lbp.U.c.U.xR, None)
        self.assertEqual(lbp.U.c.U.xZ, None)
        self.assertEqual(lbp.U.c.U.xB, None)

if __name__ == "__main__":
    unittest.main()
