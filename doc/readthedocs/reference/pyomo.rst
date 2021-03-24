Pyomo Models
============

Pyomo Representation
--------------------

Describe the SubModel component and other functionality use to 
support the Pyomo problem representation.

.. autoclass:: pao.pyomo.components.SubModel
    :special-members: __init__

.. autofunction:: pao.pyomo.convert.convert_pyomo2MultilevelProblem

PAO Solvers
-----------

.. currentmodule:: pao.pyomo.solvers.mpr_solvers

.. autoclass:: PyomoSubmodelSolver_FA
    :members:
    :inherited-members:

.. autoclass:: PyomoSubmodelSolver_REG
    :members:
    :inherited-members:

.. autoclass:: PyomoSubmodelSolver_PCCG
    :members:
    :inherited-members:
