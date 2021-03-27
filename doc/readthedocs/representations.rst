Model Representations
=====================

PAO supports *distinct* representations for algebraic models (using
`Pyomo <https://github.com/Pyomo/pyomo>`_) and compact *Multilevel
Problem Representations* (MPRs) that express objective and constraints
using vector and matrix data types.  The following sections provide
detailed descriptions of these representations that are illustrated with
increasingly complex examples:

* bilevel
* bilevel with multiple lower-levels
* trilevel
* general multilevel

Additionally, the following sections describe the setup of linear and
quadratic problems, and the transformations that can be applied to them
in PAO.

.. note::

    We do not restrict the description of PAO representations to models
    that PAO can solve. Rather, the goal of this section is to illustrate
    the breadth of the adversarial optimization problems that can be
    expressed with PAO.

Pyomo Models
------------

.. testsetup:: pyomo_repn

    import pyomo.environ as pe
    import pao
    import pao.pyomo

PAO can be used to express linear and quadratic problems in `Pyomo
<https://github.com/Pyomo/pyomo>`_ using a :class:`.SubModel`
component, which was previously introduced in the **pyomo.bilevel**
package [PyomoBookII]_.  `Pyomo <https://github.com/Pyomo/pyomo>`_
represents optimization models using an model objects that are
annotated with modeling component objects.  Thus, :class:`.SubModel`
component is a simple extension of the modeling concepts in `Pyomo
<https://github.com/Pyomo/pyomo>`_.

.. hint::

    Advanced Pyomo users will realize that the PAO :class:`.SubModel` component
    is a special case of the Pyomo ``Block`` component, which is used to
    structure the expression of Pyomo models.

A :class:`.SubModel` component creates a context for expressing the
objective and constraints in a lower-level model.  Pyomo models can
include nested and parallel :class:`.SubModel` components to express
complex multilevel problems.

Bilevel Examples
~~~~~~~~~~~~~~~~

Consider the following bilevel problem:

.. math::
   :nowrap:
 
    \begin{equation*}
    \textbf{Model PAO1}\\
    \begin{array}{ll}
    \min_{x\in[2,6],y} & x + 3 z \\
    \textrm{s.t.} & x + y = 10\\
    & \begin{array}{lll}
      \max_{z \geq 0} & z &\\
      \textrm{s.t.} & x+z &\leq 8\\
      & x + 4 z &\geq 8\\
      & x + 2 z &\leq 13
      \end{array}
    \end{array}
    \end{equation*}

This problem has has linear upper- and lower-level problems with different
objectives in each level.

.. doctest:: pyomo_repn

    >>> M = pe.ConcreteModel()

    >>> M.x = pe.Var(bounds=(2,6))
    >>> M.y = pe.Var()

    >>> M.L = pao.pyomo.SubModel(fixed=[M.x,M.y])
    >>> M.L.z = pe.Var(bounds=(0,None))

    >>> M.o = pe.Objective(expr=M.x + 3*M.L.z, sense=pe.minimize)
    >>> M.c = pe.Constraint(expr= M.x + M.y == 10)

    >>> M.L.o = pe.Objective(expr=M.L.z, sense=pe.maximize)
    >>> M.L.c1 = pe.Constraint(expr= M.x + M.L.z <= 8)
    >>> M.L.c2 = pe.Constraint(expr= M.x + 4*M.L.z >= 8)
    >>> M.L.c3 = pe.Constraint(expr= M.x + 2*M.L.z <= 13)

    >>> opt = pao.Solver("pao.pyomo.FA")
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.L.z.value)
    6.0 4.0 2.0

This example illustrates the flexibility of Pyomo representations in PAO:

* Each level can express different objectives with different senses
* Variables can be bounded or unbounded
* Equality and inequality constraints can be expressed

The :class:`.SubModel` component is used to define a logically separate
optimization model that includes variables that are dynamically fixed
by upper-level problems.  All of the Pyomo objective and constraint
declarations contained in the :class:`.SubModel` declaration are included
in the sub-problem that it defines, even if they are nested in Pyomo
``Block`` components.  The :class:`.SubModel` component also declares
which variables are fixed in a lower-level problem.  The value of the
`fixed` argument is a Pyomo variable or a list of variables.  For example,
the following model expresses the upper-level variables with a single
variable, `M.x`, which is fixed in the :class:`.SubModel` declaration:

.. doctest:: pyomo_repn

    >>> M = pe.ConcreteModel()

    >>> M.x = pe.Var([0,1])
    >>> M.x[0].setlb(2)
    >>> M.x[0].setub(6)

    >>> M.L = pao.pyomo.SubModel(fixed=M.x)
    >>> M.L.z = pe.Var(bounds=(0,None))

    >>> M.o = pe.Objective(expr=M.x[0] + 3*M.L.z, sense=pe.minimize)
    >>> M.c = pe.Constraint(expr= M.x[0] + M.x[1] == 10)

    >>> M.L.o = pe.Objective(expr=M.L.z, sense=pe.maximize)
    >>> M.L.c1 = pe.Constraint(expr= M.x[0] + M.L.z <= 8)
    >>> M.L.c2 = pe.Constraint(expr= M.x[0] + 4*M.L.z >= 8)
    >>> M.L.c3 = pe.Constraint(expr= M.x[0] + 2*M.L.z <= 13)

    >>> opt = pao.Solver("pao.pyomo.FA")
    >>> results = opt.solve(M)
    >>> print(M.x[0].value, M.x[1].value, M.L.z.value)
    6.0 4.0 2.0

Although a lower-level problem is logically a separate optimization model,
you cannot use a :class:`.SubModel` that is defined with a separate Pyomo 
model object.  Pyomo implicitly requires that all variables used in 
objective and constraint expressions are attributes of the same Pyomo model.
However, the location of variable declarations in a Pyomo model does *not* denote their 
use in upper- or lower-level problems.  For example, consider the following
model that re-expresses the previous problem:

.. doctest:: pyomo_repn

    >>> M = pe.ConcreteModel()

    >>> M.x = pe.Var(bounds=(2,6))
    >>> M.y = pe.Var()
    >>> M.z = pe.Var(bounds=(0,None))

    >>> M.o = pe.Objective(expr=M.x + 3*M.z, sense=pe.minimize)
    >>> M.c = pe.Constraint(expr= M.x + M.y == 10)

    >>> M.L = pao.pyomo.SubModel(fixed=[M.x,M.y])
    >>> M.L.o = pe.Objective(expr=M.z, sense=pe.maximize)
    >>> M.L.c1 = pe.Constraint(expr= M.x + M.z <= 8)
    >>> M.L.c2 = pe.Constraint(expr= M.x + 4*M.z >= 8)
    >>> M.L.c3 = pe.Constraint(expr= M.x + 2*M.z <= 13)

    >>> opt = pao.Solver("pao.pyomo.FA")
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.z.value)
    6.0 4.0 2.0

Note that *all* of the decision variables are declared outside of the
:class:`.SubModel` component, even though the variable ``M.z`` is a
lower-level variable.  The declarations of :class:`.SubModel` components
defines the mathematical role of all decision variables in a Pyomo model.
As this example illustrates, the specification of a bilevel problem can
be simplified if all variables are expressed at once.

Finally, we observe that PAO's Pyomo representation only works with a
subset of the many different modeling components that are supported in
`Pyomo <https://github.com/Pyomo/pyomo>`_:

* :class:`Set`
* :class:`Param`
* :class:`Var`
* :class:`Block`
* :class:`Objective`
* :class:`Constraint`

Additional Pyomo modeling components will be added to PAO as motivating
applications arise and as suitable solvers become available.

Multilevel Examples
~~~~~~~~~~~~~~~~~~~

Multilevel problems can be easily expressed with Pyomo using multiple declarations
of :class:`.SubModel`.  For example, consider the following bilevel problem that 
extends the **PAO1** model to include two equivalent lower-levels:

.. math::
   :nowrap:
 
    \begin{equation*}
    \textbf{Model PAO2}\\
    \begin{array}{ll}
    \min_{x\in[2,6],y} & x + 3 z_1 + 3 z_2 \\
    \textrm{s.t.} & x + y = 10\\
    & \begin{array}{lll}
      \max_{z_1 \geq 0} & z_1 &\\
      \textrm{s.t.} & x+z_1 &\leq 8\\
      & x + 4 z_1 &\geq 8\\
      & x + 2 z_1 &\leq 13\\
      \end{array}\\
    & \begin{array}{lll}
      \max_{z_2 \geq 0} & z_2 &\\
      \textrm{s.t.} & y+z_2 &\leq 8\\
      & y + 4 z_2 &\geq 8\\
      & y + 2 z_2 &\leq 13\\
      \end{array}\\
    \end{array}
    \end{equation*}

The **PAO2** model can be expressed in Pyomo as follows:

.. doctest:: pyomo_repn

    >>> M = pe.ConcreteModel()

    >>> M.x = pe.Var(bounds=(2,6))
    >>> M.y = pe.Var()
    >>> M.z = pe.Var([1,2], bounds=(0,None))

    >>> M.o = pe.Objective(expr=M.x + 3*M.z[1]+3*M.z[2], sense=pe.minimize)
    >>> M.c = pe.Constraint(expr= M.x + M.y == 10)

    >>> M.L1 = pao.pyomo.SubModel(fixed=[M.x])
    >>> M.L1.o = pe.Objective(expr=M.z[1], sense=pe.maximize)
    >>> M.L1.c1 = pe.Constraint(expr= M.x + M.z[1] <= 8)
    >>> M.L1.c2 = pe.Constraint(expr= M.x + 4*M.z[1] >= 8)
    >>> M.L1.c3 = pe.Constraint(expr= M.x + 2*M.z[1] <= 13)

    >>> M.L2 = pao.pyomo.SubModel(fixed=[M.y])
    >>> M.L2.o = pe.Objective(expr=M.z[2], sense=pe.maximize)
    >>> M.L2.c1 = pe.Constraint(expr= M.y + M.z[2] <= 8)
    >>> M.L2.c2 = pe.Constraint(expr= M.y + 4*M.z[2] >= 8)
    >>> M.L2.c3 = pe.Constraint(expr= M.y + 2*M.z[2] <= 13)

    >>> opt = pao.Solver("pao.pyomo.FA")
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.z[1].value, M.z[2].value)
    2.0 8.0 5.5 0.0


Multilevel Problem Representations
----------------------------------

PAO includes several *Multilevel Problem Representations*
(MPRs) that represent multilevel optimization problems with an explicit,
compact representation that simplifies the implementation of solvers
for bilevel, trilevel and other multilevel optimization problems.

For example, PAO includes a compact representation for linear bilevel
problems, ``LinearMultilevelProblem``.  Several solvers have been
developed for problems expressed as a ``LinearMultilevelProblem``,
including the big-M method proposed by Fortuny-Amat and McCarl
[FortunyMcCarl]_.  


These sections will provide a detailed discussion of the algebraic and
compact representations supported by PAO.

.. todo::
    Details about the PAO and LinearMultilevelProblem representations, showing
    the range of multi-level problems they can express.

