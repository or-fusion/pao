Solvers
=======

After formulating a multilevel problem, PAO users will generally need to
(1) transform the model to a standard form, and (2) apply an optimizer
to solve the problem.  The examples in the previous sections illustrate
that step (1) is often optional;  PAO automates the applications of
several model transformations, particularly for problems formulated with
Pyomo.  The following section summarizes the solvers available in PAO,
and describes how PAO manages solvers.  Section :ref:`transformations`
describes model transformations in PAO.

Summary of PAO Solvers
----------------------

The following summarizes the current solvers available in PAO:

* pao.mpr.FA, pao.pyomo.FA

        PAO solver for Multilevel Problem Representations that define linear
        bilevel problems.  Solver uses big-M relaxations discussed by Fortuny-
        Amat and McCarl (1981).

* pao.mpr.MIBS, pao.pyomo.MIBS

        PAO solver for Multilevel Problem Representations using the COIN-OR 
        MibS solver by Tahernejad, Ralphs, and DeNegre (2020).

* pao.mpr.PCCG, pao.pyomo.PCCG

        PAO solver for Multilevel Problem Representations that define linear
        bilevel problems. Solver uses projected column constraint generation
        algorithm described by Yue et al. (2017).

* pao.mpr.REG, pao.pyomo.REG

        PAO solver for Multilevel Problem Representations that define linear
        bilevel problems.  Solver uses regularization discussed by Scheel and
        Scholtes (2000) and Ralph and Wright (2004).

The following table summarize key features of the problems these solvers
can be applied to:

+------------------------------+-------------------------+
|                              | **Solver**              |
+------------------------------+-----+-----+------+------+
| **Problem Feature**          |*FA* |*REG*|*PCCG*|*MibS*|
+-----------------+------------+-----+-----+------+------+
|                 | Linear     | Y   | Y   | Y    | Y    |
| Equation        +------------+-----+-----+------+------+
| Structure       | Bilinear   | Y   | Y   | Y    | Y    |
|                 +------------+-----+-----+------+------+
|                 | Nonlinear  |     |     |      |      |
+-----------------+------------+-----+-----+------+------+
| Upper-Level     | Integer    | Y   |     | Y    | Y    |
| Variables       +------------+-----+-----+------+------+
|                 | Real       | Y   | Y   | Y    | Y    |
+-----------------+------------+-----+-----+------+------+
| Lower-Level     | Integer    |     |     | Y    | Y    |
| Variables       +------------+-----+-----+------+------+
|                 | Real       | Y   | Y   | Y    | Y    |
+-----------------+------------+-----+-----+------+------+
| Multilevel      | Bilevel    | Y   | Y   | Y    | Y    |
| Representation  +------------+-----+-----+------+------+
|                 | Trilevel   |     |     |      |      |
|                 +------------+-----+-----+------+------+
|                 | k-Bilevel  | Y   | Y   |      |      |
+-----------------+------------+-----+-----+------+------+

.. note::

    The iterface to MibS is a prototype that has not been well-tested.
    This interface will be documented and finalized in an upcoming 
    release of PAO.


The Solver Interface
--------------------

.. testsetup:: solver_tests

    import pyomo.environ as pe
    import pao
    import pao.pyomo

    M = pe.ConcreteModel()

    M.x = pe.Var(bounds=(2,6))
    M.y = pe.Var()

    M.L = pao.pyomo.SubModel(fixed=[M.x,M.y])
    M.L.z = pe.Var(bounds=(0,None))

    M.o = pe.Objective(expr=M.x + 3*M.L.z, sense=pe.minimize)
    M.c = pe.Constraint(expr= M.x + M.y == 10)

    M.L.o = pe.Objective(expr=M.L.z, sense=pe.maximize)
    M.L.c1 = pe.Constraint(expr= M.x + M.L.z <= 8)
    M.L.c2 = pe.Constraint(expr= M.x + 4*M.L.z >= 8)
    M.L.c3 = pe.Constraint(expr= M.x + 2*M.L.z <= 13)

The :py:obj:`.Solver` object provides a single interface for setting up
an interface to optimizers in PAO.  This includes *both* PAO solvers for
multilevel optimization problems, but also interfaces to conventional
numerical solvers that are used by PAO solvers.  We illustrate this
distinction with the following example, which optimizes the PAO1
:eq:`eq-pao1` example:

.. doctest:: solver_tests

    >>> # Create an interface to the PAO FA solver
    >>> opt = pao.Solver("pao.pyomo.FA")

    >>> # Optimize the model
    >>> # By default, FA uses the glpk MIP solver
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.L.z.value)
    6.0 4.0 2.0


    >>> # Create an interface to the PAO FA solver, using cbc
    >>> opt = pao.Solver("pao.pyomo.FA", mip_solver="cbc")

    >>> # Optimize the model using cbc
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.L.z.value)
    6.0 4.0 2.0

The :data:`.Solver` object is initialized using the solver name followed
by solver-specific options.  In this case, the FA algorithm accepts
the ``mip_solver`` option that specifies the mixed-integer programming
(MIP) solver that is used to solve the MIP that is generated by FA after
reformulating the bilevel problem.  The value of ``mip_solver`` is itself
an optimizer.  As illustrated here, this option can simply be the string
name of the MIP solver that will be used.  However, the :class:`.Solver`
object can be used to define a MIP solver interface as well:

.. doctest:: solver_tests


    >>> # Create an interface to the cbc MIP solver
    >>> mip = pao.Solver("cbc")
    >>> # Create an interface to the PAO FA solver, using cbc
    >>> opt = pao.Solver("pao.pyomo.FA", mip_solver=mip)

    >>> # Optimize the model using cbc
    >>> results = opt.solve(M)
    >>> print(M.x.value, M.y.value, M.L.z.value)
    6.0 4.0 2.0

This enables the customization of the MIP solver used by FA.  Note that
the :meth:`solve` method accepts the same options as :class:`.Solve`.
This allows for more dynamic specification of solver options:

.. doctest:: solver_tests


    >>> # Create an interface to the cbc MIP solver
    >>> cbc = pao.Solver("cbc")
    >>> # Create an interface to the glpk MIP solver
    >>> glpk = pao.Solver("glpk")

    >>> # Create an interface to the PAO FA solver
    >>> opt = pao.Solver("pao.pyomo.FA")

    >>> # Optimize the model using cbc
    >>> results = opt.solve(M, mip_solver=cbc)
    >>> print(M.x.value, M.y.value, M.L.z.value)
    6.0 4.0 2.0

    >>> # Optimize the model using glpk
    >>> results = opt.solve(M, mip_solver=glpk)
    >>> print(M.x.value, M.y.value, M.L.z.value)
    6.0 4.0 2.0

.. warning::

    The :meth:`solve` current passes unknown keyword arguments to the
    optimizer used by PAO solvers, but this feature will be disabled.


PAO Solvers
~~~~~~~~~~~

Solvers developed in PAO have names that begin with ``pao.``.
The current set of available PAO solvers can be queried using the
:class:`.Solver` object:

.. doctest:: solver_tests

    >>> for name in pao.Solver:
    ...     print(name)
    pao.mpr.FA
    pao.mpr.MIBS
    pao.mpr.PCCG
    pao.mpr.REG
    pao.pyomo.FA
    pao.pyomo.MIBS
    pao.pyomo.PCCG
    pao.pyomo.REG

    >>> pao.Solver.summary()
    pao.mpr.FA
        PAO solver for Multilevel Problem Representations that define linear
        bilevel problems.  Solver uses big-M relaxations discussed by Fortuny-
        Amat and McCarl (1981).
    <BLANKLINE>
    pao.mpr.MIBS
        PAO solver for Multilevel Problem Representations using the COIN-OR
        MibS solver by Tahernejad, Ralphs, and DeNegre (2020).
    <BLANKLINE>
    pao.mpr.PCCG
        PAO solver for Multilevel Problem Representations that define linear
        bilevel problems. Solver uses projected column constraint generation
        algorithm described by Yue et al. (2017).
    <BLANKLINE>
    pao.mpr.REG
        PAO solver for Multilevel Problem Representations that define linear
        bilevel problems.  Solver uses regularization discussed by Scheel and
        Scholtes (2000) and Ralph and Wright (2004).
    <BLANKLINE>
    pao.pyomo.FA
        PAO solver for Pyomo models that define linear and bilinear bilevel
        problems.  Solver uses big-M relaxations discussed by Fortuny-Amat and
        McCarl (1981).
    <BLANKLINE>
    pao.pyomo.MIBS
        PAO solver for Multilevel Problem Representations using the COIN-OR
        MibS solver by Tahernejad, Ralphs, and DeNegre (2020).
    <BLANKLINE>
    pao.pyomo.PCCG
        PAO solver for Pyomo models that define linear and bilinear bilevel
        problems.  Solver uses projected column constraint generation
        algorithm described by Yue et al. (2017)
    <BLANKLINE>
    pao.pyomo.REG
        PAO solver for Pyomo models that define linear and bilinear bilevel
        problems.  Solver uses regularization discussed by Scheel and Scholtes
        (2000) and Ralph and Wright (2004).
    <BLANKLINE>

The :meth:`solve` method includes documentation describing the keyword
arguments for a specific solver.  For example:

.. doctest:: solver_tests

    >>> opt = pao.Solver("pao.pyomo.FA")
    >>> help(opt.solve)
    Help on method solve in module pao.pyomo.solvers.mpr_solvers:
    <BLANKLINE>
    solve(model, **options) method of pao.pyomo.solvers.mpr_solvers.PyomoSubmodelSolver_FA instance
        Executes the solver and loads the solution into the model.
    <BLANKLINE>
        Parameters
        ----------
        model
            The model that is being optimized.
        options
            Keyword options that are used to configure the solver.
    <BLANKLINE>
        Keyword Arguments
        -----------------
        tee
          If True, then solver output is streamed to stdout. (default is False)
        load_solutions
          If True, then the finale solution is loaded into the model. (default is True)
        linearize_bigm
          The name of the big-M value used to linearize bilinear terms.  If this is not specified, then the solver will throw an error if bilinear terms exist in the model.
        mip_solver
          The MIP solver used by FA.  (default is glpk)
    <BLANKLINE>
        Returns
        -------
        Results
            A summary of the optimization results.
    <BLANKLINE>

..  ***

The :meth:`solve` method returns a results object that contains
data about the optimization process.  In particular, this object
contains information about the termination conditions for the solver.
The :meth:`check_optimal_termination` method can be used confirm that the
termination condition indicates that an optimal solution was found.  For example:

.. doctest:: solver_tests

    >>> nlp = pao.Solver('ipopt', print_level=3)
    >>> opt = pao.Solver('pao.pyomo.REG', nlp_solver=nlp)
    >>> results = opt.solve(M) #doctest:+ELLIPSIS
    W...
    >>> print(results.solver.termination_condition)
    TerminationCondition.optimal
    >>> results.check_optimal_termination()
    True
 
Pyomo Solvers
~~~~~~~~~~~~~

The :class:`.Solver` object also provides a convenient interface to
conventional numerical solvers.  Currently, solver objects constructed
by :class:`.Solver` are simple wrappers around Pyomo optimization
solver objects.  This interface supports two types of solver
interfaces: (1) solvers that execute locally, and (2) solvers that execute
on remote servers.

When optimizating a **Pyomo** model, solver parameters can be setup
both when the solver interface is created and when a model is optimized.
For example:

.. doctest:: solver_tests

    >>> # This is a nonlinear toy problem modeled with Pyomo
    >>> NLP = pe.ConcreteModel()
    >>> A = list(range(10))
    >>> NLP.x = pe.Var(A, bounds=(0,None), initialize=1)
    >>> NLP.o = pe.Objective(expr=sum(pe.sin((i+1)*NLP.x[i]) for i in A))
    >>> NLP.c = pe.Constraint(expr=sum(NLP.x[i] for i in A) >= 1)

    >>> nlp = pao.Solver('ipopt', print_level=3)
    >>> # Apply ipopt with print level 3
    >>> results = nlp.solve(NLP)
    >>> # Override the default print level to using 5
    >>> results = nlp.solve(NLP, print_level=5)

However, PAO users will typically setup solver parameters when the 
Pyomo solver is initially created:

.. doctest:: solver_tests

    >>> nlp = pao.Solver('ipopt', print_level=3)
    >>> opt = pao.Solver('pao.pyomo.REG', nlp_solver=nlp)
    >>> results = opt.solve(M) #doctest:+ELLIPSIS
    W...

When executing locally, the :keyword:`executable` option can be used
to explicitly specify the path to the executable that is used by this solver.
This is helpful in contexts where Pyomo is not automatically finding the *correct* 
optimizer executable in a user's shell environment.

When executing on a remote server, the :keyword:`server` is used to
specify the server that is used.  Currently, only the ``neos`` server is
supported, which allows the user to perform optimization at NEOS [NEOS]_.
The NEOS server requires a user to specify a valid email address:

.. code-block::

    >>> nlp = pao.Solver('ipopt', server='neos', email='pao@gmail.com')
    >>> opt = pao.Solver('pao.pyomo.REG', nlp_solver=nlp)
    >>> results = opt.solve(M)


.. warning::

    There is no common reference for solver-specific parameters for the
    solvers available in Pyomo.  These are generally documented with
    solver documentation, and users should expect to contact solver
    developers to learn about these.
