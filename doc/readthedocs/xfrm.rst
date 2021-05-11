.. _transformations:

Model Transformations
=====================

PAO includes a variety of functions that transform models, which generally 
are applied as follows:

.. code-block::

    >>> new_model, soln = transform_function(old_model)

Here, the function ``transform_func`` generates the model ``new_model`` from the model ``old_model``.  The
object ``soln`` is used to map a solution back to the old model:

.. code-block::

    >>> soln.copy(From=new_model, To=old_model)

The following transformation functions are documented and suitable for use by end-users:

* :func:`pao.pyomo.convert.convert_pyomo2MultilevelProblem`

    This function generates a :class:`.LinearMultilevelProblem` or
    :class:`.QuadraticMultilevelProblem` from a Pyomo model.  By default,
    all constraints in the MPR representation are inequalities.

* :func:`pao.mpr.convert_repn.linearize_bilinear_terms`

    This function generates a :class:`.LinearMultilevelProblem` from a :class:`.QuadraticMultilevelProblem`
    that only contains bilinear terms.  This transformation currently is limited to 
    MPRs that only contain inequality constraints.

* :func:`pao.mpr.convert_repn.convert_to_standard_form`

    This function generates an equivalent linear multilevel representation for which all
    variables are non-negative and all constraints have the same form (inequalities or equalities).
    This simplifies the implementation of solvers, which typically assume a standard form
    for subproblems.

