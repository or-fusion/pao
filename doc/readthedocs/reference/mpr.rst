Compact Models
==============

LinearMultilevelProblem Representation
--------------------------------------

.. todo::
    Flush out the description of the LinearMultilevelProblem, QuadraticMultilevelProblem and other functionality use to 
    support compact problem representations.

.. currentmodule:: pao.mpr.repn

.. autoclass:: LinearMultilevelProblem
   :members:

.. autoclass:: QuadraticMultilevelProblem
   :members:

Model Transformations
---------------------

.. currentmodule:: pao.mpr.convert_repn

.. autofunction:: convert_to_standard_form

.. autofunction:: linearize_bilinear_terms

PAO Solvers
-----------

.. autoclass:: pao.mpr.solvers.fa.LinearMultilevelSolver_FA
    :members:
    :inherited-members:

.. autoclass:: pao.mpr.solvers.reg.LinearMultilevelSolver_REG
    :members:
    :inherited-members:

.. autoclass:: pao.mpr.solvers.pccg.LinearMultilevelSolver_PCCG
    :members:
    :inherited-members:


