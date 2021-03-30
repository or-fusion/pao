.. PAO documentation master file, created by
   sphinx-quickstart on Sat Jan 11 18:36:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />



PAO Documentation
=================

PAO is a Python-based package for Adversarial Optimization.  The goal of
this package is to provide a general modeling and analysis capability for
bilevel, trilevel and other multilevel optimization forms that express
adversarial dynamics.  PAO integrates two different modeling abstractions:

1. **Algebraic models** extend the modeling concepts in the
   `Pyomo <https://github.com/Pyomo/pyomo>`_ algebraic modeling language
   to express problems with an intuitive algebraic syntax.  Thus, we
   expect that this modeling abstraction will commonly be used by PAO
   end-users.

2. **Compact models** express objective and constraints in a manner
   that is typically used to express the mathematical form of these
   problems (e.g. using vector and matrix data types).  PAO defines
   custom *Multilevel Problem Representations* (MPRs) that simplify the
   implementation of solvers for bilevel, trilevel and other multilevel 
   optimization problems.


.. toctree::
   :maxdepth: 1

   installation.rst
   overview.rst
   examples.rst
   representations/pyomo.rst
   representations/mpr.rst
   solvers.rst
   xfrm.rst
   reference.rst
   bibliography.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


PAO Resources
-------------

PAO development is hosted at GitHub:

* https://github.com/or-fusion/pao

The OR-Fusion GitHub organization is used to coordinate installation of
PAO with other OR-related capabilities:

* https://github.com/or-fusion

Ask a question on StackOverflow:

* https://stackoverflow.com/questions/ask?tags=pyomo

