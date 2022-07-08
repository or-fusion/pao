import unittest
import pyomo.environ as pe
from pyomo.contrib import appsi
from pao.duality.dual import Dual
from itertools import product
from pyomo.core.base.var import ScalarVar


class TestNonlinearDual(unittest.TestCase):
    def test_linear(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var(bounds=(-10, 10))
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + 2)
        m.c2 = pe.Constraint(expr=m.y >= -m.x)

        opt = appsi.solvers.Gurobi()
        res1 = opt.solve(m)

        d = Dual().dual(m)
        res2 = opt.solve(d)

        self.assertAlmostEqual(
            res1.best_feasible_objective,
            res2.best_feasible_objective
        )

    def test_nonlinear(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)

        opt = appsi.solvers.Ipopt(only_child_vars=False)
        res1 = opt.solve(m)

        d = Dual().dual(m)
        res2 = opt.solve(d)

        self.assertAlmostEqual(
            res1.best_feasible_objective,
            res2.best_feasible_objective
        )

    def test_equality(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 2))
        m.y = pe.Var(bounds=(0, 2))

        m.obj = pe.Objective(expr=-2*m.x - 3*m.y)
        m.c1 = pe.Constraint(expr=m.x + m.y == 2)
        m.c2 = pe.Constraint(expr=-2*m.x - 4*m.y >= -6)

        opt1 = appsi.solvers.Gurobi(only_child_vars=False)
        opt2 = appsi.solvers.Gurobi(only_child_vars=False)

        pd = Dual()
        d = pd.dual(m)

        res1 = opt1.solve(m)
        res2 = opt2.solve(d)

        self.assertAlmostEqual(
            res1.best_feasible_objective,
            res2.best_feasible_objective
        )

        del m.c1
        m.c1 = pe.Constraint(expr=m.x + m.y == 1)
        d = pd.dual(m)
        res1 = opt1.solve(m)
        res2 = opt2.solve(d)
        self.assertAlmostEqual(
            res1.best_feasible_objective,
            res2.best_feasible_objective
        )

    def test_persistent_binary_branching(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 2))
        m.y = pe.Var(bounds=(0, 2))

        m.obj = pe.Objective(expr=-2*m.x - 3*m.y)
        m.c1 = pe.Constraint(expr=m.x + m.y <= 2)
        m.c2 = pe.Constraint(expr=2*m.x + 4*m.y <= 6)

        opt1 = appsi.solvers.Gurobi(only_child_vars=False)
        opt2 = appsi.solvers.Gurobi(only_child_vars=False)

        opt1.gurobi_options['DualReductions'] = 0
        opt2.gurobi_options['DualReductions'] = 0

        pd = Dual()
        d = pd.dual(m)

        res1 = opt1.solve(m)
        res2 = opt2.solve(d)

        self.assertAlmostEqual(
            res1.best_feasible_objective,
            res2.best_feasible_objective
        )

        opt1.config.load_solution = False
        opt2.config.load_solution = False

        for xval, yval in product([None, 0, 1, 2], [None, 0, 1, 2]):
            if xval is None:
                m.x.unfix()
            else:
                m.x.fix(xval)

            if yval is None:
                m.y.unfix()
            else:
                m.y.fix(yval)

            res1 = opt1.solve(m)
            d = pd.dual(m)
            res2 = opt2.solve(d)

            if res1.termination_condition == appsi.base.TerminationCondition.infeasible:
                self.assertEqual(
                    res2.termination_condition,
                    appsi.base.TerminationCondition.unbounded
                )
            else:
                self.assertIsNotNone(res1.best_feasible_objective)
                self.assertAlmostEqual(
                    res1.best_feasible_objective,
                    res2.best_feasible_objective
                )

        m.x.unfix()
        m.y.unfix()

        res1 = opt1.solve(m)
        d = pd.dual(m)
        res2 = opt2.solve(d)

        self.assertIsNotNone(res1.best_feasible_objective)
        self.assertAlmostEqual(
            res1.best_feasible_objective,
            res2.best_feasible_objective
        )

    def test_persistent_changing_bounds(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)

        opt1 = appsi.solvers.Ipopt(only_child_vars=False)
        opt2 = appsi.solvers.Ipopt(only_child_vars=False)

        pd = Dual()

        def check():
            d = pd.dual(m)
            res1 = opt1.solve(m)
            res2 = opt2.solve(d)
            self.assertAlmostEqual(
                res1.best_feasible_objective,
                res2.best_feasible_objective
            )

        check()

        m.c1.deactivate()
        check()

        m.x.setlb(1)
        check()

        m.x.setlb(0.1)
        check()

        m.x.setlb(None)
        m.x.setub(2)
        check()

        m.x.setub(0.1)
        check()

        m.x.setub(None)
        check()

        m.c1.activate()
        check()

    def test_fixed_vars(self):
        m = pe.ConcreteModel()
        m.x = ScalarVar(bounds=(0, 2))
        m.y = ScalarVar(bounds=(0, 2))
        m.c = ScalarVar(bounds=(-3, 3))

        m.obj = pe.Objective(expr=-2*m.x - m.c*m.y)
        m.c1 = pe.Constraint(expr=m.x + m.y <= 2)
        m.c2 = pe.Constraint(expr=-2*m.c*m.x + 4*m.y <= 6)

        opt = appsi.solvers.Gurobi(only_child_vars=False)
        opt.gurobi_options['nonconvex'] = 2

        pd = Dual(fixed_vars=[m.c])
        d = pd.dual(m)

        res = opt.solve(d)
        self.assertAlmostEqual(res.best_feasible_objective, -2)
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y.value, None)
        self.assertAlmostEqual(m.c.value, -3)
