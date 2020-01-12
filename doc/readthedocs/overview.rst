Overview
------------

This is draft ReadTheDocs documentation for PAO.

Many planning situtations involve the analysis of several objectives
that reflect a hierarchy of decision-makers.  For example, policy
decisions are made at different levels of a government, each of which
has a different objective and decision space.  Similarly, robust planning
against adversaries is often modeled with a 2-level hierarchy, where the
defensive planner makes decisions that account for adversarial response.

PAO extends Pyomo's optimization modeling environment to express
multilevel optimization applications.  Multilevel optimization techniques
partition control over decision variables amongst the levels.  Decisions
at each level of the hierarchy may be constrained by decisions at other
levels, and the objectives for each level may account for decisions made
at other levels.  Thus, the goal of PAO is to express this structure in a
manner that is intuitive to users, and which facilitates the application
of appropriate optimization solvers.

PAO includes modeling capabilities originally developed within
`pyomo.bilevel`. Pyomo users observed that `pyomo.bilevel` failed to meet
either of these two goals particularly well, so this project is taking
a step back to reassess the design of `pyomo.bilevel` to better express
multilevel optimization problems and to facilitate the development of
suitable optimization solvers using intuitive problem representations.

