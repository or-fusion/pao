.. PAO documentation master file, created by
   sphinx-quickstart on Sat Jan 11 18:36:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PAO Documentation | DRAFT
=========================

PAO is a Python-based package for Adversarial Optimization.  The goal of
this package is to provide a general modeling and analysis capability for
bilevel, trilevel and other optimization forms that express adversarial
dynamics.  PAO integrates two different modeling abstractions:

* *Tailored models* that express objective and constraints in a manner that is typically used to express the mathematical form of these problems (e.g. using vector and matrix data types).

* *General algebraic models* that extend the modeling concepts in the `Pyomo <https://github.com/Pyomo/pyomo>`_ algebraic modeling language to express problems with a simple algebraic syntax.


.. toctree::
   :maxdepth: 2

   installation.rst
   overview.rst

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

* https://stackoverflow.com/questions/ask?tags=pyomo-community

