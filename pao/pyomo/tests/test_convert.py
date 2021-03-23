import numpy as np
import pyutilib.th as unittest
import pyomo.environ as pe
from pao.pyomo.convert import collect_multilevel_tree, convert_pyomo2LinearMultilevelProblem, convert_pyomo2MultilevelProblem
from pao.pyomo import SubModel
from pao.mpr import LinearMultilevelProblem, QuadraticMultilevelProblem


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

    def test5a(self):
        # All variables used in an expression
        # One of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.lb = pe.Param(initialize=0)
        M.ub = pe.Param(initialize=1)
        M.x = pe.Var()
        M.y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.z[4].setlb(M.lb)
        M.z[4].setub(M.ub)
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
        M.o = pe.Objective(expr=1+2*M.x+3*sum(M.y[i] for i in M.y)+4*sum(M.z[i] for i in M.z))
        
        M.s = SubModel(fixed=M.x)
        M.s.o = pe.Objective(expr=5+6*M.x+7*sum(M.y[i] for i in M.y)+8*sum(M.z[i] for i in M.z))

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=True)
        
        U = lmpr.U
        L = lmpr.U.LL

        self.assertEqual(len(U.x), len(M.x))
        self.assertEqual(len(L.x), len(M.z)+len(M.y))

        self.assertEqual(U.d, 1)
        self.assertEqual(list(U.c[U]), [2])
        self.assertEqual(list(U.c[L]), [4,4,3,3,4])

        self.assertEqual(L.d, 5)
        self.assertEqual(list(L.c[U]), [6])
        self.assertEqual(list(L.c[L]), [8,8,7,7,8])

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

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=True)
        
        U = lmpr.U
        L = lmpr.U.LL

        self.assertEqual(len(U.x), len(M.x)+len(M.z)+len(M.y))
        self.assertEqual(U.d, 0)

        self.assertEqual(list(U.c[U]), [1,3,3,3,2,2])

    def test_initialize_2L(self):
        # All variables used in an expression
        # None of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.p = pe.Param(initialize=1)
        M.x = pe.Var([0])
        M.X = pe.Var([0])
        M.y = pe.Var([1,2], within=pe.Binary)
        M.Y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.Z = pe.Var([3,4,5], within=pe.Integers)
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

        M.s.o = pe.Objective(expr=M.X[0]+2*sum(M.Y[i] for i in M.Y)+3*sum(M.Z[i] for i in M.Z))

        M.s.c1 = pe.Constraint([0], rule=c_rule)
        M.s.c2 = pe.Constraint(expr=M.x[0] <= np.PINF)
        M.s.c3 = pe.Constraint(expr=M.p <= 2)

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=True)
        
        U = lmpr.U
        L = lmpr.U.LL

        self.assertEqual(len(U.x), len(M.x)+len(M.z)+len(M.y))
        self.assertEqual(len(L.x), len(M.X)+len(M.Z)+len(M.Y))

        self.assertEqual(U.d, 0)

        self.assertEqual(list(U.c[U]), [1,3,3,3,2,2])
        self.assertEqual(U.c[L], None)

        self.assertEqual(L.c[U], None)
        self.assertEqual(list(L.c[L]), [1,3,3,3,2,2])

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
        M.c = pe.Constraint(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z) == 0)
        
        M.s = SubModel(fixed=M.x)
        #M.s.c = pe.Constraint(expr=M.x[0]+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z) == 0)

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=True)
        
        U = lmpr.U
        L = lmpr.U.LL

        self.assertEqual(len(U.x), len(M.x)+len(M.z)+len(M.y))

        self.assertEqual(U.c[U], None)

        # 2 rows because the lmpr is an inequality, so the equality is split in two
        self.assertEqual(U.A[U].shape, (2,6))
        self.assertEqual([list(U.A[U].toarray()[i]) for i in range(2)], [[1,3,3,3,2,2], [-1,-3,-3,-3,-2,-2]])

    def test_initialize_3a(self):
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
        M.c = pe.Constraint(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z) == 0)
        
        M.s = SubModel(fixed=M.x)
        #M.s.c = pe.Constraint(expr=M.x[0]+sum(M.y[i] for i in M.y)+sum(M.z[i] for i in M.z) == 0)

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=False)
        
        U = lmpr.U
        L = lmpr.U.LL

        self.assertEqual(len(lmpr.U.x), len(M.x)+len(M.z)+len(M.y))

        self.assertEqual(U.c[U], None)

        # 2 rows because the lmpr is an inequality, so the equality is split in two
        self.assertEqual(U.A[U].shape, (1,6))
        self.assertEqual([list(U.A[U].toarray()[i]) for i in range(1)], [[1,3,3,3,2,2]])

    def test_initialize_4a(self):
        # All variables used in an expression
        # None of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.x = pe.Var([0])
        M.X = pe.Var([0])
        M.y = pe.Var([1,2], within=pe.Binary)
        M.Y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.Z = pe.Var([3,4,5], within=pe.Integers)
        M.x[0].setlb(0)
        M.x[0].setub(1)
        M.z[4].setlb(0)
        M.z[4].setub(2)
        M.o = pe.Objective(expr=1)
        M.c = pe.Constraint(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z) == 0)
        M.C1 = pe.Constraint(expr=M.x[0] >= 0)
        M.C2 = pe.Constraint(expr=M.x[0] <= 0)
        M.C3 = pe.Constraint(expr=pe.inequality(0, M.x[0], 1))
        
        M.s = SubModel(fixed=M.x)
        M.s.c = pe.Constraint(expr=M.X[0]+sum(M.Y[i] for i in M.Y)+sum(M.Z[i] for i in M.Z) == 0)

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=True)
        
        U = lmpr.U
        L = U.LL

        self.assertEqual(len(U.x), len(M.x)+len(M.z)+len(M.y))
        self.assertEqual(len(L.x), len(M.X)+len(M.Z)+len(M.Y))

        self.assertEqual(U.c[U], None)

        # 2 rows because the lmpr is an inequality, so the equality is split in two
        self.assertEqual(U.A[U].shape, (6,6))
        self.assertEqual([list(U.A[U].toarray()[i]) for i in range(6)],
[[1.0,3,3,3,2,2], [-1.0,-3,-3,-3,-2,-2], [-1.0,0,0,0,0,0], [1.0,0,0,0,0,0], [-1.0,0,0,0,0,0], [1.0,0,0,0,0,0]])

        # 2 rows because the lmpr is an inequality, so the equality is split in two
        self.assertEqual(L.A[L].shape, (2,6))
        self.assertEqual([list(L.A[L].toarray()[i]) for i in range(2)], [[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]])

    def test_initialize_4b(self):
        # All variables used in an expression
        # None of the Integer variables looks like a binary
        M = pe.ConcreteModel()
        M.x = pe.Var([0])
        M.X = pe.Var([0])
        M.y = pe.Var([1,2], within=pe.Binary)
        M.Y = pe.Var([1,2], within=pe.Binary)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.Z = pe.Var([3,4,5], within=pe.Integers)
        M.x[0].setlb(0)
        M.x[0].setub(1)
        M.z[4].setlb(0)
        M.z[4].setub(2)
        M.o = pe.Objective(expr=1)
        M.c = pe.Constraint(expr=M.x[0]+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z) == 0)
        M.C1 = pe.Constraint(expr=M.x[0] >= 0)
        M.C2 = pe.Constraint(expr=M.x[0] <= 0)
        M.C3 = pe.Constraint(expr=pe.inequality(0, M.x[0], 1))
        
        M.s = SubModel(fixed=M.x)
        M.s.c = pe.Constraint(expr=M.X[0]+sum(M.Y[i] for i in M.Y)+sum(M.Z[i] for i in M.Z) == 0)

        lmpr,_ = convert_pyomo2LinearMultilevelProblem(M, inequalities=False)
        
        U = lmpr.U
        L = U.LL

        self.assertEqual(len(U.x), len(M.x)+4+len(M.z)+len(M.y))
        self.assertEqual(len(L.x), len(M.X)+len(M.Z)+len(M.Y))

        self.assertEqual(U.c[U], None)

        # 2 rows because the lmpr is an inequality, so the equality is split in two
        self.assertEqual(U.A[U].shape, (5,10))
        self.assertEqual([list(U.A[U].toarray()[i]) for i in range(5)],
[[ 1.0, 0.0, 0.0, 0.0, 0.0, 3, 3, 3, 2, 2],
 [-1.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0],
 [ 1.0, 0.0, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0],
 [-1.0, 0.0, 0.0, 1.0, 0.0, 0, 0, 0, 0, 0],
 [ 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0]])

        # 2 rows because the lmpr is an inequality, so the equality is split in two
        self.assertEqual(L.A[L].shape, (1,6))
        self.assertEqual([list(L.A[L].toarray()[i]) for i in range(1)], [[1,1,1,1,1,1]])

    def test_initialize_6(self):
        # Bilinear terms in objective
        M = pe.ConcreteModel()
        M.x = pe.Var(within=pe.Binary)
        M.y = pe.Var([1,2], within=pe.Reals)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.o = pe.Objective(expr=2*M.z[3] + 3*M.x*M.y[1])
        M.c = pe.Constraint(expr=M.x+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z) <= 0)
        
        M.s = SubModel(fixed=M.x)
        M.s.o = pe.Objective(expr=4*M.z[3] + 3*M.y[1] - 5*M.x*M.y[2])

        qmp,_ = convert_pyomo2MultilevelProblem(M)
        #qmp.print()
        
        self.assertEqual(type(qmp), QuadraticMultilevelProblem)
        U = qmp.U
        L = qmp.U.LL

        self.assertEqual(len(U.x), 3)
        self.assertEqual(len(L.x), 3)

        self.assertEqual(U.c[U], None)
        self.assertEqual(list(U.c[L]), [0,0,2])
        self.assertEqual(U.P[U,L].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        self.assertEqual(L.c[U], None)
        self.assertEqual(list(L.c[L]), [3,0,4])
        self.assertEqual(L.P[U,L].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -5.0, 0.0]])

        self.assertEqual(U.A[U].toarray().tolist(),
[[3.0, 3.0, 1.0]])
        self.assertEqual(U.A[L].toarray().tolist(), [[2,2,3]])

        self.assertEqual(L.A[U], None)
        self.assertEqual(L.A[L], None)

    def test_initialize_7(self):
        # Bilinear terms in objective
        M = pe.ConcreteModel()
        M.x = pe.Var(within=pe.Binary)
        M.y = pe.Var([1,2], within=pe.Reals)
        M.z = pe.Var([3,4,5], within=pe.Integers)
        M.o = pe.Objective(expr=2*M.z[3] + 3*M.x*M.y[1])
        M.c = pe.Constraint(expr=M.x+2*sum(M.y[i] for i in M.y)+3*sum(M.z[i] for i in M.z) + 7*M.x*M.y[1] <= 0)
        
        M.s = SubModel(fixed=M.x)
        M.s.o = pe.Objective(expr=4*M.z[3] + 3*M.y[1] - 5*M.x*M.y[2])
        M.s.c = pe.Constraint(expr=11*M.x*M.z[3] == 0)

        qmp,_ = convert_pyomo2MultilevelProblem(M)
        #qmp.print()
        
        self.assertEqual(type(qmp), QuadraticMultilevelProblem)
        U = qmp.U
        L = qmp.U.LL

        self.assertEqual(len(U.x), 3)
        self.assertEqual(len(L.x), 3)

        self.assertEqual(U.c[U], None)
        self.assertEqual(list(U.c[L]), [0,0,2])
        self.assertEqual(U.P[U,L].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        self.assertEqual(L.c[U], None)
        self.assertEqual(list(L.c[L]), [3,0,4])
        self.assertEqual(L.P[U,L].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -5.0, 0.0]])

        self.assertEqual(U.A[U].toarray().tolist(),
[[3.0, 3.0, 1.0]])
        self.assertEqual(U.A[L].toarray().tolist(), [[2,2,3]])
        self.assertEqual(U.Q[U,L][0].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])

        self.assertEqual(L.A[U], None)
        self.assertEqual(L.A[L], None)
        self.assertEqual(L.Q[U,L][0].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 11.0]])
        self.assertEqual(L.Q[U,L][1].toarray().tolist(),
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -11.0]])

if __name__ == "__main__":
    unittest.main()
