Pyomo Models
============

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
   :label: eq-pao1
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

* :class:`Set` - Set declarations
* :class:`Param` - Parameter declarations
* :class:`Var` - Variable declarations
* :class:`Block` - Defines a subset of a model
* :class:`Objective` - Define a model objective
* :class:`Constraint` - Define model constraints

Additional Pyomo modeling components will be added to PAO as motivating
applications arise and as suitable solvers become available.

Multilevel Examples
~~~~~~~~~~~~~~~~~~~

Multilevel problems can be easily expressed with Pyomo using multiple declarations
of :class:`.SubModel`.

Multiple Lower Levels
^^^^^^^^^^^^^^^^^^^^^

Consider the following bilevel problem that 
extends the **PAO1** model to include two equivalent lower-levels:

.. math::
   :label: eq-pao2
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

Trilevel Problems
^^^^^^^^^^^^^^^^^

Trilevel problems can be described with nested declarations of :class:`.SubModel` components.  Consider the 
following trilevel continuous linear problem described by Anadalingam [Anadalingam]:

.. math::
   :label: eq-anadalingam
   :nowrap:
 
    \begin{equation*}
    \textbf{Model Anadalingam1988}\\
    \begin{array}{llll}
    \min_{x_1 \geq 0} & -7 x_1 - 3 x_2 + 4 x_3 \\
    \textrm{s.t.} & \min_{x_2 \geq 0} & -x_2 \\
                  & \textrm{s.t.} & \min_{x_3 \in [0,0.5]} & -x_3 \\
                  &               & \textrm{s.t.} & x_1 + x_2 + x_3 \leq 3\\
                  &               &               & x_1 + x_2 - x_3 \leq 1\\
                  &               &               & x_1 + x_2 + x_3 \geq 1\\
                  &               &               & -x_1 + x_2 + x_3 \leq 1\\
    \end{array}
    \end{equation*}

This model can be expressed in Pyomo as follows:

.. doctest:: pyomo_repn

    >>> M = pe.ConcreteModel()
    >>> M.x1 = pe.Var(bounds=(0,None))
    >>> M.x2 = pe.Var(bounds=(0,None))
    >>> M.x3 = pe.Var(bounds=(0,0.5))

    >>> M.L = pao.pyomo.SubModel(fixed=M.x1)

    >>> M.L.B = pao.pyomo.SubModel(fixed=M.x2)

    >>> M.o = pe.Objective(expr=-7*M.x1 - 3*M.x2 + 4*M.x3)

    >>> M.L.o = pe.Objective(expr=-M.x2)
    >>> M.L.B.o = pe.Objective(expr=-M.x3)

    >>> M.L.B.c1 = pe.Constraint(expr=   M.x1 + M.x2 + M.x3 <= 3)
    >>> M.L.B.c2 = pe.Constraint(expr=   M.x1 + M.x2 - M.x3 <= 1)
    >>> M.L.B.c3 = pe.Constraint(expr=   M.x1 + M.x2 + M.x3 >= 1)
    >>> M.L.B.c4 = pe.Constraint(expr= - M.x1 + M.x2 + M.x3 <= 1)

.. note::

    PAO solvers cannot currently solve trilevel solvers like this,
    but an issue has been submitted to add this functionality.

Bilinear Problems
^^^^^^^^^^^^^^^^^

PAO models using Pyomo represent general quadratic problems with quadratic
terms in the objective and constraints at each level.  The special case
where bilinear terms arise with an upper-level binary variable multiplied
with a lower-level variable is common in many applications.  For this case, the PAO solvers
for Pyomo models include an option to linearize these bilinear terms.

The following models considers a variation of the **PAO1** model where binary variables control
the expression of lower-level constraints:

.. math::
   :nowrap:
 
    \begin{equation*}
    \textbf{Model PAO3}\\
    \begin{array}{ll}
    \min_{x\in[2,6],y,w_1,w_2} & x + 3 z + 5 w_1\\
    \textrm{s.t.} & x + y = 10\\
    & w_1 + w_2 \geq 1\\
    & w_1,w_2 \in \{0,1\}\\
    & \begin{array}{lll}
      \max_{z \geq 0} & z &\\
      \textrm{s.t.} & x+ w_1 z &\leq 8\\
      & x + 4 z &\geq 8\\
      & x + 2 w_2 z &\leq 13
      \end{array}
    \end{array}
    \end{equation*}

The **PAO3** model can be expressed in Pyomo as follows:

.. doctest:: pyomo_repn

    >>> M = pe.ConcreteModel()

    >>> M.w = pe.Var([1,2], within=pe.Binary)
    >>> M.x = pe.Var(bounds=(2,6))
    >>> M.y = pe.Var()
    >>> M.z = pe.Var(bounds=(0,None))

    >>> M.o = pe.Objective(expr=M.x + 3*M.z+5*M.w[1], sense=pe.minimize)
    >>> M.c1 = pe.Constraint(expr= M.x + M.y == 10)
    >>> M.c2 = pe.Constraint(expr= M.w[1] + M.w[2] >= 1)

    >>> M.L = pao.pyomo.SubModel(fixed=[M.x,M.y,M.w])
    >>> M.L.o = pe.Objective(expr=M.z, sense=pe.maximize)
    >>> M.L.c1 = pe.Constraint(expr= M.x + M.w[1]*M.z <= 8)
    >>> M.L.c2 = pe.Constraint(expr= M.x + 4*M.z >= 8)
    >>> M.L.c3 = pe.Constraint(expr= M.x + 2*M.w[2]*M.z <= 13)

    >>> opt = pao.Solver("pao.pyomo.FA", linearize_bigm=100)
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.z.value, M.w[1].value, M.w[2].value)
    6.0 4.0 3.5 0 1

