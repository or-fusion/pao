Simple Examples
===============

We illustrate the PAO algebraic and compact representations with a series
of simple examples.  Consider the following bilevel problem introduced
by Bard [Bard98]_ (example 5.1.1):

.. math::
   :nowrap:

    \begin{equation*}
    \begin{array}{lll}
    \min_{x \geq 0} & x - 4 y & \\
    \textrm{s.t.} & \min_{y \geq 0} & y\\
    & \textrm{s.t.} & -x -y \leq -3\\
    & & -2 x + y \leq 0\\
    & & 2 x + y \leq 12\\
    & & 3 x - 2 y \leq 4
    \end{array}
    \end{equation*}

This problem has has linear upper- and lower-level problems with different
objectives in each level.  Thus, this problem can be represented in
PAO using a Pyomo model representation and the ``LinearBilevelProblem``
representation.

Using Pyomo
-----------

The following python script defines a bilevel problem in Pyomo:

.. code-block:: python

    import pyomo.environ as pe
    from pao.bilevel import *

    M = pe.ConcreteModel()
    M.x = pe.Var(bounds=(0,None))
    M.y = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.x - 4*M.y)

    M.L = SubModel(fixed=M.x)
    M.L.o = pe.Objective(expr=M.y)
    M.L.c1 = pe.Constraint(expr=   -M.x -   M.y <= -3)
    M.L.c2 = pe.Constraint(expr= -2*M.x +   M.y <=  0)
    M.L.c3 = pe.Constraint(expr=  2*M.x +   M.y <= 12)
    M.L.c4 = pe.Constraint(expr=  3*M.x - 2*M.y <=  4)

    with SolverFactory('pao.submodel.FA') as solver:
        results = solver.solve(M)

The ``SubModel`` component defines a Pyomo block object within which the
lower-level problem is declared.  The ``fixed`` option is used to specify
the upper-level variables whose value is fixed in the lower-level problem.

The ``pao.submodel.FA`` uses the method for solving linear bilevel
programs with big-M relaxations described by Fortuny-Amat and McCarl
[FortunyMcCarl]_.  By deault, the final values of the upper- and
lower-level variables are loaded back into the Pyomo model.  The
``results`` object contains information about the problem, the solver
and the solver status.


Using LinearBilevelProblem
--------------------------

Bilevel linear problems can also be represented using the
``LinearBilevelProblem`` class.  This class provides a simple mechanism
for organizing data for variables, objectives and linear constraints.  The following
examples illustrate the use of ``LinearBilevelProblem`` for Bard's example 5.1.1 described
above, but this representation can naturally be used to express multi-level problems as well
as problems with multiple lower-levels.

Using Numpy and Scipy Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following python script defines a bilevel problem using ``LinearBilevelProblem`` with
numpy and scipy data:

.. code-block:: python

    import numpy as np
    from scipy.sparse import coo_matrix
    from pao.lbp import *

    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = np.array([0])
    U.c[U] = np.array([1])
    U.c[L] = np.array([-4])

    L.x.lower_bounds = np.array([0])

    L.c[L] = np.array([1])
    L.A[U] = coo_matrix((np.array([-1, -2, 2, 3]),
                        (np.array([0, 1, 2, 3]),
                         np.array([0, 0, 0, 0]))))
    L.A[L] = coo_matrix((np.array([-1, 1, 1, -2]),
                        (np.array([0, 1, 2, 3]),
                         np.array([0, 0, 0, 0]))))
    L.b = np.array([-3, 0, 12, 4])

    with SolverFactory('pao.lbp.FA') as solver:
        results = solver.solve(M)

The ``U`` and ``L`` objects represent the upper- and lower-level
respectively.  When declaring these objects, the user specifies the number
of real, integer and binary variables.  The remaining declarations assume
that these variables are used in that order.  Thus, there is a single
declaration for the objective coefficients, ``c``, which is an array
with values for each of the declared variables.  However, the upper-
and lower-level objective coefficients are separately declared for
the upper- and lower-level variables by indexing ``c`` with ``U`` and
``L`` respectively.  This example includes declarations for the upper-
and lower-level variable bounds and objective coefficients.  There are no
upper-level constraints, so only the lower-level constriants are declared.

Note that the syntax for specifying solvers is directly analogous to that
used with Pyomo models.  The same solver options are available.  The only
difference is the specification of the solver name that indicates the
expected type of the model that will be solved.

Using Python Lists and Dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the constraint matrices are dense, the ``coo_matrix``
is used to illustrate the general support for sparse data.  The
``LinearBilevelProblem`` class also supports a simpler syntax where
dense arrays can be specified and Python lists and sparse matrices can
be specified with Python tuple and dictionary objects:

.. code-block:: python

    from pao.lbp import *

    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = [0]
    U.c[U] = [1]
    U.c[L] = [-4]

    L.x.lower_bounds = [0]

    L.c[L] = [1]
    L.A[U] = (4,1), {(0,0):-1, (1,0):-2, (2,0): 2, (3,0): 3}
    L.A[L] = (4,1), {(0,0):-1, (1,0): 1, (2,0): 1, (3,0):-2}
    L.b = [-3, 0, 12, 4]

    with SolverFactory('pao.lbp.FA') as solver:
        results = solver.solve(M)

When specifying a sparse matrix, a tuple is provided.  The first element is a 2-tuple that
defines the shape of the matrix, and the second element is a dictionary that defines the
non-zero values in the sparse matrix.

Similarly, a list-of-lists syntax can be used to specify dense matrices:

.. code-block:: python

    from pao.lbp import *

    M = LinearBilevelProblem()

    U = M.add_upper(nxR=1)
    L = U.add_lower(nxR=1)

    U.x.lower_bounds = [0]
    U.c[U] = [1]
    U.c[L] = [-4]

    L.x.lower_bounds = [0]

    L.c[L] = [1]
    L.A[U] = [[-1], [-2], [-2], [3]]
    L.A[L] = [[-1], [1], [1], [-2]]
    L.b = [-3, 0, 12, 4]

    with SolverFactory('pao.lbp.FA') as solver:
        results = solver.solve(M)


When native Python data values are used to initialize a
``LinearBilevelProblem``, they are converted into numpy and scipy
data types.  This facilitates the use of ``LinearBilevelProblem`` objects for defining
numerical solvers using a consistent, convenient API for numerical operations (e.g. matrix-vector
multiplication).
