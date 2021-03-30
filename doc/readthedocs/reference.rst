Library Reference
=================

The following classes and functions represent the core functionality
in PAO:

.. toctree::
   :maxdepth: 1

   reference/solverapi.rst
   reference/pyomo.rst
   reference/mpr.rst

.. warning::

    The logic in ``pao.duality`` is currently disabled.  There are known errors in this code
    that will be resolved by re-implementing it using the logic in ``pao.mpr``.
