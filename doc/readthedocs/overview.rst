Overview
------------

Many planning situations involve the analysis of several objectives
that reflect a hierarchy of decision-makers.  For example, policy
decisions are made at different levels of a government, each of which
has a different objective and decision space.  Similarly, robust planning
against adversaries is often modeled with a 2-level hierarchy, where the
defensive planner makes decisions that account for adversarial response.
Multilevel optimization techniques partition control over decision
variables amongst the levels.  Decisions at each level of the hierarchy
may be constrained by decisions at other levels, and the objectives for
each level may account for decisions made at other levels.

The goal of PAO is to express this structure in a manner that is
intuitive to users, and which facilitates the application of appropriate
optimization solvers.  Another key goal of PAO is to facilite the
development of new solvers.  However, we recognize that the data
structures used to represent algebraic models suitable for users are
poorly suited for complex numerical algorithms.  Consequently, PAO
supports *separate* representations for algebraic models (using Pyomo)
and explicit, compact problem representations (e.g. using numpy and
scipy data).  Additionally, PAO automates the translation of algebraic
models to compact representations to leverage the solvers developed for 
compact representations.

For example, PAO includes a compact representation for linear bilevel
problems, LinearBilevelProblem.  Several solvers have been developed for
problems expressed as a LinearBilevelProblem, including the big-M method
proposed by Fortuny-Amat and McCarl [CITE HERE].  PAO can similarly
express linear bilevel problems in Pyomo using a Submodel component,
which was previously introduced in the **pyomo.bilevel** package.
Further, these Pyomo models can be automatically converted to the compact
LinearBilevelProblem representation and solved with solvers tailored for that
problem.

TODO - Figure illustrating the use of Pyomo and LinearBilevelProblem representations.

The use of independent problem representations in this manner has
several implications for PAO.  First, this design facilitates the
development of solvers for algebraic modeling languages like Pyomo
that are intrinsically more robust.  We have observed that compact
representations like LinearBilevelProblem enable the development of
solvers using natural operations (e.g. matrix-vector multiplication).
Thus, we argue that these solvers will be more robust and easier to
maintain when compared to solvers developed using Pyomo data structures
(e.g. expression trees).  Additionally, the conversion of a Pyomo representation
to a compact representation provides a context for verifying that the Pyomo
model represents the intended problem form.

Second, this design facilitates the development of different problem
representations that may or may not be inter-operable.  Although PAO is
derived from initial efforts with **pyomo.bilevel**, it has evolved from
an extension of Pyomo's modeling capability to be a library of capabilites
that are synergistic with Pyomo but not strictly dependent on it.

