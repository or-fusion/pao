import unittest
import pyomo.environ as pe
from pyomo.contrib import appsi
from pao.duality.kkt import construct_kkt
import coramin


class TestKKT(unittest.TestCase):
    def test_linear(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var(bounds=(-10, 10))
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + 2)
        m.c2 = pe.Constraint(expr=m.y >= -m.x)

        opt1 = appsi.solvers.Gurobi()
        res1 = opt1.solve(m)
        self.assertEqual(
            res1.termination_condition,
            appsi.base.TerminationCondition.optimal
        )

        d = construct_kkt(m)
        opt2 = appsi.solvers.Gurobi()
        opt2.gurobi_options['nonconvex'] = 2
        m.obj.sense = pe.maximize
        res2 = opt2.solve(d)
        self.assertEqual(
            res2.termination_condition,
            appsi.base.TerminationCondition.optimal
        )

        res1 = res1.solution_loader.get_primals()
        res2 = res2.solution_loader.get_primals()

        for k, v in res1.items():
            self.assertAlmostEqual(v, res2[k], 5)

    def test_milp(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 2), domain=pe.Integers)
        m.y = pe.Var(bounds=(0, 2), domain=pe.Integers)

        m.obj = pe.Objective(expr=-2*m.x - 3*m.y)
        m.c1 = pe.Constraint(expr=m.x + m.y <= 2)
        m.c2 = pe.Constraint(expr=2*m.x + 4*m.y <= 6)

        opt1 = appsi.solvers.Gurobi()
        res1 = opt1.solve(m)
        self.assertEqual(
            res1.termination_condition,
            appsi.base.TerminationCondition.optimal
        )

        d = construct_kkt(m)
        opt2 = appsi.solvers.Gurobi()
        opt2.gurobi_options['nonconvex'] = 2
        m.obj.sense = pe.maximize
        res2 = opt2.solve(d)
        self.assertEqual(
            res2.termination_condition,
            appsi.base.TerminationCondition.optimal
        )

        res1 = res1.solution_loader.get_primals()
        res2 = res2.solution_loader.get_primals()

        for k, v in res1.items():
            self.assertAlmostEqual(v, res2[k], 5)

    def test_nonlinear(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)

        opt1 = appsi.solvers.Ipopt()
        res1 = opt1.solve(m)

        d = construct_kkt(m)
        opt2 = appsi.solvers.Ipopt()
        m.obj.sense = pe.maximize
        res2 = opt2.solve(d)

        self.assertEqual(
            res1.termination_condition,
            appsi.base.TerminationCondition.optimal
        )
        self.assertEqual(
            res2.termination_condition,
            appsi.base.TerminationCondition.optimal
        )

        res1 = res1.solution_loader.get_primals()
        res2 = res2.solution_loader.get_primals()

        for k, v in res1.items():
            self.assertAlmostEqual(v, res2[k])


if __name__ == '__main__':
    unittest.main()
