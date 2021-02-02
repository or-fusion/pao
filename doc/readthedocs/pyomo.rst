Pyomo Models
============

Pyomo Representation
--------------------

Describe the SubModel component and other functionality use to 
support the Pyomo problem representation.

.. autoclass:: pao.bilevel.components.SubModel
    :special-members: __init__

.. autofunction:: pao.bilevel.convert.convert_pyomo2LinearBilevelProblem

PAO Solvers
-----------

.. currentmodule:: pao.bilevel.solvers.lbp_solvers

.. autoclass:: PyomoSubmodelSolver_FA
    :members:
    :inherited-members:

.. autoclass:: PyomoSubmodelSolver_REG
    :members:
    :inherited-members:

.. autoclass:: PyomoSubmodelSolver_PCCG
    :members:
    :inherited-members:
