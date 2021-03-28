PAO Solvers and Model Transformations
=====================================

After formulating a multilevel problem, PAO users will generally need to
(1) transform the model to a standard form, and (2) apply an optimizer
to solve the problem.  The examples in the previous sections illustrate
that step (1) is often optional.  PAO automates the applications of
several model transformations, particularly for problems formulated
with Pyomo.  The following section describes how PAO manages solvers, and how model transformations can be applied in sequence.

The Solver Interface
--------------------

.. testsetup:: pyomo_repn

    import pyomo.environ as pe
    import pao
    import pao.pyomo

The :class:`.Solver` class provides a single interface for setting up an interface to optimizers in PAO.  This includes *both* PAO solvers for multilevel optimization problems, but also interfaces to conventional numerical solvers that are used by PAO solvers.  We illustrate this distinction with the following example:

.. doctest::




.. todo::

    Give worked examples for each of the solvers.

    Discuss solver options.

    Discuss termination conditions and error handling.

    Describe interfaces to Pyomo solvers.

    Provide pointers for solver-specific parameters
