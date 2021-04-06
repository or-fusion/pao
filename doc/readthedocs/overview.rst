Overview
========

Many planning situations involve the analysis of a hierarchy of
decision-makers with competing objectives.  For example, policy decisions
are made at different levels of a government, each of which has a
different objective and decision space.  Similarly, robust planning
against adversaries is often modeled with a 2-level hierarchy, where
the defensive planner makes decisions that account for adversarial
response.  Multilevel optimization techniques partition control over
decision variables amongst the levels.  Decisions at each level of
the hierarchy may be constrained by decisions at other levels, and the
objectives for each level may account for decisions made at other levels.
The PAO library is designed to express this structure in a manner that is
intuitive to users, and which facilitates the application of appropriate
optimization solvers.  In particular, PAO extends the modeling concepts
in the `Pyomo <https://github.com/Pyomo/pyomo>`_ algebraic modeling
language to express problems with an intuitive algebraic syntax.

However, data structures used to represent algebraic models are often
poorly suited for complex numerical algorithms.  Consequently, PAO
supports *distinct* representations for algebraic models (using `Pyomo
<https://github.com/Pyomo/pyomo>`_) and compact problem representations
that express objective and constraints using vector and matrix data types.
Currently, PAO includes several *Multilevel Problem Representations*
(MPRs) that represent multilevel optimization problems with an explicit,
compact representation that simplifies the implementation of solvers
for bilevel, trilevel and other multilevel optimization problems.

For example, PAO includes a compact representation for linear bilevel
problems, ``LinearMultilevelProblem``.  Several solvers have been
developed for problems expressed as a ``LinearMultilevelProblem``,
including the big-M method proposed by Fortuny-Amat and McCarl
[FortunyMcCarl]_.  PAO can similarly express linear bilevel problems in
Pyomo using a ``Submodel`` component, which was previously introduced in
the **pyomo.bilevel** package [PyomoBookII]_.  Further, these Pyomo models
can be automatically converted to the compact ``LinearMultilevelProblem``
representation and optimized with solvers tailored for that representation.

The use of independent problem representations in this manner has
several implications for PAO.  First, this design facilitates the
development of solvers for algebraic modeling languages like Pyomo
that are intrinsically more robust.  Compact representations like
``LinearMultilevelProblem`` enable the development of solvers using
natural operations (e.g. matrix-vector multiplication).  Thus, we expect
these solvers to be more robust and easier to maintain when compared to
solvers developed using Pyomo data structures (e.g. expression trees).
Additionally, the conversion of a Pyomo representation to a compact
representation provides a context for verifying that the Pyomo model
represents the intended problem form.

Second, this design facilitates the development of different problem
representations that may or may not be inter-operable.  Although PAO
is derived from initial efforts with **pyomo.bilevel**, it has evolved
from an extension of Pyomo's modeling capability to be a library that
is synergistic with Pyomo but not strictly dependent on it.

The following sections provide detailed descriptions of these
representations that are illustrated with increasingly complex examples,
including

* bilevel
* bilevel with multiple lower-levels
* trilevel

Additionally, the following sections describe the setup of linear and
quadratic problems, and the transformations that can be applied to them
in PAO.

.. note::

    We do not restrict the description of PAO representations to
    models that PAO can solve. Rather, a goal of this documentation
    to illustrate the breadth of the adversarial optimization problems
    that can be expressed with PAO, thereby motivating the development
    of new solvers for new classes of problems.

.. note::

    Although the modeling abstractions in PAO are suitable for both
    pessimistic and optimistic multilevel optimization formulations,
    PAO currently assumes that all problems are expressed and solved
    in an an optimistic form.  Thus, when there multiple solutions in
    lower-level problems, the choice of the solution is determined by
    the follower not the leader.  This convention reflects the fact that
    solution techniques for pessimistic forms are not well-developed.

