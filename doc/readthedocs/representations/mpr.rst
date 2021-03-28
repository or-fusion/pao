Multilevel Representations
==========================

.. testsetup:: mpr_repn

    import numpy as np
    import pao
    import pao.mpr

PAO includes several *Multilevel Problem Representations* (MPRs)
that represent multilevel optimization problems with an explicit,
compact representation that simplifies the implementation of solvers for
bilevel, trilevel and other multilevel optimization problems.  These MPRs
express objective and constraints using vector and matrix data types.
However, they organize these data types to provide a semantically clear
organization of multilevel problems.  Additionally, the MPRs provide
checks to ensure the consistency of the data within and across levels.

The classes :class:`.LinearMultilevelProblem`
and :class:`.QuadraticMultilevelProblem` respectively
represent linear and quadratic multilevel problems.  Although
:class:`.QuadraticMultilevelProblem` is a generalization of the
representation in :class:`.LinearMultilevelProblem`, the use of tailored
representations for different classes of problems clarifies the semantic
context when using them.  For example, this allows for simple error checking
to confirm that a problem is linear.

Currently, all PAO solvers for MPRs support only linear problems, so the
following sections focus on :class:`.LinearMultilevelProblem`.  However,
we conclude with an example of model transformations that enable the
solution of quadratic problems using :class:`.QuadraticMultilevelProblem`.

.. note::

    We do not expect many users to directly employ a MPR data
    representation for their applications.  Perhaps this would be
    desirable if their problem was already represented with matrix and
    vector data.  In general, the algebraic representation supported by
    Pyomo will be more convenient for large, complex applications.

    We expect this representation to be more useful for researchers
    developing multilevel solvers, since the MPR representations provide
    structure that simplifies the expression of necessary mathematical
    operations for these problems.


Linear Bilevel Examples
~~~~~~~~~~~~~~~~~~~~~~~

We consider again the bilevel problem PAO1 :eq:`eq-pao1`.  This problem
has has linear upper- and lower-level problems with different objectives
in each level.

.. doctest:: mpr_repn

    >>> M = pao.mpr.LinearMultilevelProblem()

    >>> U = M.add_upper(nxR=2)
    >>> L = U.add_lower(nxR=1)

    >>> U.x.lower_bounds = [2, np.NINF]
    >>> U.x.upper_bounds = [6, np.PINF]
    >>> L.x.lower_bounds = [0]
    >>> L.x.upper_bounds = [np.PINF]

    >>> U.c[U] = [1, 0]
    >>> U.c[L] = [3]

    >>> L.c[L] = [1]
    >>> L.maximize = True

    >>> U.equalities = True
    >>> U.A[U] = [[1, 1]]
    >>> U.b = [10]

    >>> L.A[U] = [[ 1, 0],
    ...          [-1, 0],
    ...          [ 1, 0]]
    >>> L.A[L] = [[ 1],
    ...          [-4],
    ...          [ 2]]
    >>> L.b = [8, -8, 13]

    >>> M.check()

    >>> opt = pao.Solver("pao.mpr.FA")
    >>> results = opt.solve(M)
    >>> print(M.U.x.values)
    [6.0, 4.0]
    >>> print(M.U.LL.x.values)
    [2.0]

The example illustrates both the flexibility of the MPR representions
in PAO but also the structure they enforce on the multilevel problem
representation.  The upper-level problem is created by calling the
:meth:`.add_upper` method, which takes arguments that specify the
variables at that level:

* ``nxR`` - The number of real variables (Default: 0)
* ``nxZ`` - The number of general integer variables (Default: 0)
* ``nxB`` - The number of binary variables (Default: 0)

In each level, the variables are represented as a vector of values,
ordered in this manner.

Similarly, the :meth:`.add_lower` method is used to generate a lower-level
problem from a given level.  Note that this allows for the specification
of arbitrary nesting of levels, since a lower-level can be defined
relative to any other level in the model.  Additionally, multiple
lower-levels can be specified for relative to a single level (see below).

The :meth:`.add_upper` and :meth:`.add_lower` methods return the
corresponding level object, which is used to specify data in the model
later.

For a given level object, ``Z``, the data ``Z.x`` contains
information about the decision variables.  In particular, the values
``Z.x.lower_bounds`` and ``Z.x.upper_bounds`` can be set with arrays
of numeric values to specify lower- and upper-bounds on the decision
variables.  Note that missing lower- and upper-bounds are specified with
``numpy.NINF`` and ``numpy.PINF`` respectively.

The ``Z.c`` data specifies coefficients of the objective function for
this level.  This data is indexed by a level object ``B`` to indicate
the data associated with the variables in ``B``.  In the example above:

* ``U.c[U]`` is the array of coefficients of the upper-level objective for the variables in the upper-level,
* ``U.c[L]`` is the array of coefficients of the upper-level objective for the variables in the lower-level, and
* ``L.c[L]`` is the array of coefficients of the lower-level objective for the variables in the lower-level.

Since ``L.c[U]`` is not specified, it has a value ``None`` that
indicates that no upper-level variables have non-zero coefficients in the
lower-level objective.  The ``Z.A`` data specifies the matrix coefficients
for the constraints using a similar indexing notation and semantics.

The values ``Z.minimize`` and ``Z.maximize`` can be set to ``True``
to indicate whether the objective in ``Z`` minimizes or maximizes.
(The default is minimize.)  Similarly the value ``Z.inequalities``
and ``Z.equalities`` can be set to ``True`` to indicate whether the
constraints in ``Z`` are inequalities or equalities.  (The default
is inequalities.)  Finally, the value ``Z.b`` defines the array of
constraint right-hand-side values.

The :meth:``check`` method provides a convenient sanity check that the
data is defined consistently within each level and between levels.

Note that PAO supports a consistent interface for creating a solver
interface and for applying solvers.  In fact, the user should be
aware that Pyomo and MPR solvers are named in a consistent fashion.
For example, the Pyomo solver **pao.pyomo.FA** calls the MPR solver
**pao.mpr.FA** after automatically converting the Pyomo representation
to a :class:``LinearMultilevelProblem`` representation.  This example
illustrates that values ``Z.x.values`` contains the values of each level
``Z`` after optimization.

Multilevel Examples
~~~~~~~~~~~~~~~~~~~

Multilevel problems can be easily expressed using the same MPR data 
representation.

Multiple Lower Levels
^^^^^^^^^^^^^^^^^^^^^

We consider again the bilevel problem PAO2 :eq:`eq-pao2`, which has has
multiple lower-level problems.  The **PAO2** model can be expressed as
a linear multilevel problem as follows:

.. doctest:: mpr_repn

    >>> M = pao.mpr.LinearMultilevelProblem()
  
    >>> U = M.add_upper(nxR=2)
    >>> L1 = U.add_lower(nxR=1)
    >>> L2 = U.add_lower(nxR=1)

    >>> U.x.lower_bounds = [2, np.NINF]
    >>> U.x.upper_bounds = [6, np.PINF]
    >>> U.c[U] = [1, 0]
    >>> U.c[L1] = [3]
    >>> U.c[L2] = [3]
    >>> U.equalities = True
    >>> U.A[U] = [[1, 1]]
    >>> U.b = [10]

    >>> L1.x.lower_bounds = [0]
    >>> L1.x.upper_bounds = [np.PINF]
    >>> L1.c[L1] = [1]
    >>> L1.maximize = True
    >>> L1.A[U] = [[ 1, 0],
    ...           [-1, 0],
    ...           [ 1, 0]]
    >>> L1.A[L1] = [[ 1],
    ...            [-4],
    ...            [ 2]]
    >>> L1.b = [8, -8, 13]

    >>> L2.x.lower_bounds = [0]
    >>> L2.x.upper_bounds = [np.PINF]
    >>> L2.c[L2] = [1]
    >>> L2.maximize = True
    >>> L2.A[U] = [[0,  1],
    ...           [0, -1],
    ...           [0,  1]]
    >>> L2.A[L2] = [[ 1],
    ...            [-4],
    ...            [ 2]]
    >>> L2.b = [8, -8, 13]

    >>> opt = pao.Solver("pao.mpr.FA")
    >>> results = opt.solve(M)
    >>> print(U.x.values)
    [2.0, 8.0]
    >>> print(L1.x.values)
    [5.5]
    >>> print(L2.x.values)
    [0.0]

The declarataion of the two lower level problems is naturally contained
within the data of the ``L1`` and ``L2`` objects.  Further, the
cross-level interactions are intuitively represented using the index
notation for the objective and constraint data objects.

Note that this more explicit representation clarifies some ambiguity in
the expression of lower-levels in the Pyomo representation.  The Pyomo
representation of PAO2 only specifies the fixed variables that are
**used** in each of the two lower-level problems.  PAO analyzes
the use of decision variables in Pyomo models, and treats *unused*
variables as fixed.  Thus, the Pyomo and MPR representations generate
a consistent interpretation of the variable specifications.  However,
the MPR representation is more explicit in this regard.


Trilevel Problems
^^^^^^^^^^^^^^^^^

We consider again the trilevel problem described by Anadalingam
:eq:`eq-anadalingam`, which can be expressed as a trilevel linear problem
as follows:

.. doctest:: mpr_repn

    >>> M = pao.mpr.LinearMultilevelProblem()

    >>> U = M.add_upper(nxR=1)
    >>> U.x.lower_bounds = [0]

    >>> L = U.add_lower(nxR=1)
    >>> L.x.lower_bounds = [0]

    >>> B = L.add_lower(nxR=1)
    >>> B.x.lower_bounds = [0]
    >>> B.x.upper_bounds = [0.5]

    >>> U.minimize = True
    >>> U.c[U] = [-7]
    >>> U.c[L] = [-3]
    >>> U.c[B] = [4]

    >>> L.minimize = True
    >>> L.c[L] = [-1]

    >>> B.minimize = True
    >>> B.c[B] = [-1]
    >>> B.inequalities = True
    >>> B.A[U] = [[1], [ 1], [-1], [-1]]
    >>> B.A[L] = [[1], [ 1], [-1], [ 1]]
    >>> B.A[B] = [[1], [-1], [-1], [ 1]]
    >>> B.b = [3,1,-1,1]


Bilinear Problems
^^^^^^^^^^^^^^^^^

The :class:`.QuadraticMultilevelProblem` class can represent general
quadratic problems with quadratic terms in the objective and constraints
at each level.  The special case where bilinear terms arise with an
upper-level binary variable multiplied with a lower-level variable is
common in many applications.  For this case, PAO provides a function to
linearize these bilinear terms.

We consider again the bilevel problem PAO3 :eq:`eq-pao3`, which is 
represented and solved as follows:

.. doctest:: mpr_repn

    >>> M = pao.mpr.QuadraticMultilevelProblem()

    >>> U = M.add_upper(nxR=2, nxB=2)
    >>> L = U.add_lower(nxR=1)

    >>> U.x.lower_bounds = [2, np.NINF, 0, 0]
    >>> U.x.upper_bounds = [6, np.PINF, 1, 1]
    >>> L.x.lower_bounds = [0]
    >>> L.x.upper_bounds = [np.PINF]

    >>> U.c[U] = [1, 0, 5, 0]
    >>> U.c[L] = [3]

    >>> L.c[L] = [1]
    >>> L.maximize = True

    >>> U.A[U] = [[ 1,  1,  0,  0],
    ...          [-1, -1,  0,  0],
    ...          [ 0,  0, -1, -1]
    ...          ]
    >>> U.b = [10, -10, -1]

    >>> L.A[U] = [[ 1, 0, 0, 0],
    ...          [-1, 0, 0, 0],
    ...          [ 1, 0, 0, 0]]
    >>> L.A[L] = [[ 0],
    ...          [-4],
    ...          [ 0]]
    >>> L.Q[U,L] = (3,4,1), {(0,2,0):1, (2,3,0):2}
                
    >>> L.b = [8, -8, 13]

    >>> lmr, soln = pao.mpr.linearize_bilinear_terms(M, 100)
    >>> opt = pao.Solver("pao.mpr.FA")
    >>> results = opt.solve(lmr)
    >>> soln.copy(From=lmr, To=M)
    >>> print(U.x.values)
    [6.0, 4.0, 0, 1]
    >>> print(L.x.values)
    [3.5]

The data ``L.Q[U,L]`` specifies the bilinear terms multiplying
variables from level ``U`` with variables from level ``L``, which
are included in the constraints in level ``L``.  Note that ``Q`` is a
tensor, which is indexed over the constraints, upper-level variables
and lower-level variables.  A similar syntax is used to define bilinear
terms in objectives, ``P``, though that is represented as a sparse matrix.
Quadratic terms can be specified simply by using the same levels to index
``Q`` or ``P``.

Model transformations like :func:``.linearize_bilinear_terms`` are
described in further detail in the next section.  Note that this function
returns both the transformed model as well as a helper class that maps
solutions back to the original model.  This logic facilitates the
automation of model transformations within PAO.

