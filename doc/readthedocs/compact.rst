Compact Models
==============

LinearBilevelProblem Representation
-----------------------------------

.. todo::
    Describe the LinearBilevelProblem, QuadraticBilevelProblem and other functionality use to 
    support compact problem representations.

.. currentmodule:: pao.lbp.repn

.. autoclass:: LinearBilevelProblem
   :members:

.. currentmodule:: pao.lbp.convert_repn

.. autofunction:: convert_LinearBilevelProblem_to_standard_form

PAO Solvers
-----------

.. autoclass:: pao.lbp.solvers.fa.LinearBilevelSolver_FA
    :members:
    :inherited-members:

.. autoclass:: pao.lbp.solvers.reg.LinearBilevelSolver_REG
    :members:
    :inherited-members:

.. autoclass:: pao.lbp.solvers.pccg.LinearBilevelSolver_PCCG
    :members:
    :inherited-members:



