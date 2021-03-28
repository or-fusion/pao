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
PAO using a Pyomo model representation or the ``LinearMultilevelProblem``
representation.

Using Pyomo
-----------

The following python script defines a bilevel problem in Pyomo:

.. doctest::

    >>> import pyomo.environ as pe
    >>> from pao.pyomo import *

    # Create a model object
    >>> M = pe.ConcreteModel()

    # Define decision variables
    >>> M.x = pe.Var(bounds=(0,None))
    >>> M.y = pe.Var(bounds=(0,None))

    # Define the upper-level objective
    >>> M.o = pe.Objective(expr=M.x - 4*M.y)

    # Create a SubModel component to declare a lower-level problem
    # The variable M.x is fixed in this lower-level problem
    >>> M.L = SubModel(fixed=M.x)

    # Define the lower-level objective
    >>> M.L.o = pe.Objective(expr=M.y)

    # Define lower-level constraints
    >>> M.L.c1 = pe.Constraint(expr=   -M.x -   M.y <= -3)
    >>> M.L.c2 = pe.Constraint(expr= -2*M.x +   M.y <=  0)
    >>> M.L.c3 = pe.Constraint(expr=  2*M.x +   M.y <= 12)
    >>> M.L.c4 = pe.Constraint(expr=  3*M.x - 2*M.y <=  4)

    # Create a solver and apply it
    >>> with Solver('pao.pyomo.FA') as solver:
    ...     results = solver.solve(M)

    # The final solution is loaded into the model 
    >>> print(M.x.value)
    4.0
    >>> print(M.y.value)
    4.0

The ``SubModel`` component defines a Pyomo block object within which the
lower-level problem is declared.  The ``fixed`` option is used to specify
the upper-level variables whose value is fixed in the lower-level problem.

The ``pao.pyomo.FA`` uses the method for solving linear bilevel
programs with big-M relaxations described by Fortuny-Amat and McCarl
[FortunyMcCarl]_.  By default, the final values of the upper- and
lower-level variables are loaded back into the Pyomo model.  The
``results`` object contains information about the problem, the solver
and the solver status.


Using LinearMultilevelProblem
-----------------------------

Bilevel linear problems can also be represented using the
``LinearMultilevelProblem`` class.  This class provides a simple mechanism
for organizing data for variables, objectives and linear constraints.  The following
examples illustrate the use of ``LinearMultilevelProblem`` for Bard's example 5.1.1 described
above, but this representation can naturally be used to express multi-level problems as well
as problems with multiple lower-levels.

Using Numpy and Scipy Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following python script defines a bilevel problem using ``LinearMultilevelProblem`` with
numpy and scipy data:

.. doctest::

    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> from pao.mpr import *

    # Create a model object
    >>> M = LinearMultilevelProblem()

    # Declare the upper- and lower-levels, including the number of decision-variables
    #  nxR=1 means there will be 1 real-valued decision variable
    >>> U = M.add_upper(nxR=1)
    >>> L = U.add_lower(nxR=1)

    # Declare the bounds on the decision variables
    >>> U.x.lower_bounds = np.array([0])
    >>> L.x.lower_bounds = np.array([0])

    # Declare the upper-level objective
    #   U.c[X] is the array of coefficients in the objective for variables in level X
    >>> U.c[U] = np.array([1])
    >>> U.c[L] = np.array([-4])

    # Declare the lower-level objective, which has no upper-level decision-variables
    >>> L.c[L] = np.array([1])

    # Declare the lower-level constraints
    #   L.A[X] is the matrix coefficients in the constraints for variables in level X
    >>> L.A[U] = coo_matrix((np.array([-1, -2, 2, 3]),
    ...                    (np.array([0, 1, 2, 3]),
    ...                     np.array([0, 0, 0, 0]))))
    >>> L.A[L] = coo_matrix((np.array([-1, 1, 1, -2]),
    ...                    (np.array([0, 1, 2, 3]),
    ...                     np.array([0, 0, 0, 0]))))

    # Declare the constraint right-hand-side
    #   By default, constraints are inequalities, so these are upper-bounds
    >>> L.b = np.array([-3, 0, 12, 4])

    # Create a solver and apply it
    >>> with Solver('pao.mpr.FA') as solver:
    ...    results = solver.solve(M)

    # The final solution is loaded into the model 
    >>> print(U.x.values[0])
    4.0
    >>> print(L.x.values[0])
    4.0

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

Note that the syntax for specifying solvers is analogous to that used
with Pyomo models.  The same solver options are available.  The principle
difference is the specification of the solver name that indicates the
expected type of the model that will be solved.

Using Python Lists and Dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the constraint matrices are dense in this example, the
``coo_matrix`` is used to illustrate the general support for sparse data.
The ``LinearMultilevelProblem`` class also supports a simpler syntax
where dense arrays can be specified and Python lists and sparse matrices
can be specified with Python tuple and dictionary objects:

.. doctest::

    >>> from pao.mpr import *
    
    >>> M = LinearMultilevelProblem()
    
    >>> U = M.add_upper(nxR=1)
    >>> L = U.add_lower(nxR=1)
    
    >>> U.x.lower_bounds = [0]
    >>> L.x.lower_bounds = [0]
    
    >>> U.c[U] = [1]
    >>> U.c[L] = [-4]
    >>> L.c[L] = [1]
    
    >>> L.A[U] = (4,1), {(0,0):-1, (1,0):-2, (2,0):2, (3,0): 3}
    >>> L.A[L] = (4,1), {(0,0):-1, (1,0): 1, (2,0):1, (3,0):-2}
    
    >>> L.b = [-3, 0, 12, 4]
    
    >>> with Solver('pao.mpr.FA') as solver:
    ...    results = solver.solve(M)
    
    >>> print(U.x.values[0])
    4.0
    >>> print(L.x.values[0])
    4.0

When specifying a sparse matrix, a tuple is provided (e.g. for
``L.A[U]``).  The first element is a 2-tuple that defines the shape
of the matrix, and the second element is a dictionary that defines the
non-zero values in the sparse matrix.

Similarly, a list-of-lists syntax can be used to specify dense matrices:

.. doctest::

    >>> from pao.mpr import *

    >>> M = LinearMultilevelProblem()

    >>> U = M.add_upper(nxR=1)
    >>> L = U.add_lower(nxR=1)

    >>> U.x.lower_bounds = [0]
    >>> L.x.lower_bounds = [0]

    >>> U.c[U] = [1]
    >>> U.c[L] = [-4]
    >>> L.c[L] = [1]

    >>> L.A[U] = [[-1], [-2], [2], [3]]
    >>> L.A[L] = [[-1], [1], [1], [-2]]
    >>> L.b = [-3, 0, 12, 4]

    >>> with Solver('pao.mpr.FA') as solver:
    ...    results = solver.solve(M)

    >>> print(U.x.values[0])
    4.0
    >>> print(L.x.values[0])
    4.0

When native Python data values are used to initialize a
``LinearMultilevelProblem``, they are converted into numpy and scipy
data types.  This facilitates the use of ``LinearMultilevelProblem``
objects for defining numerical solvers using a consistent, convenient
API for numerical operations (e.g. matrix-vector multiplication).

