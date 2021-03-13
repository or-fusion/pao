Installation
============

PAO currently supports the following versions of Python:

* CPython: 3.6, 3.7, 3.8


Using GIT
---------

PAO can be installed by cloning the PAO software repostory and 
then directly installing the software.  For example, the *master*
branch can be installed as follows:

::
   
   git clone https://github.com/or-fusion/pao.git
   cd pao
   python setup.py develop

Using CONDA
-----------

Coming soon.

Using PIP
---------

The standard utility for installing Python packages is *pip*.  You
can use *pip* to install from the PAO software repository.  For
example, the *master* branch can be installed as follows:

::

    python -m pip install https://github.com/or-fusion/pao.git

Coming soon: installation from PyPI.

.. comment
   The standard utility for installing Python packages is *pip*.  You
   can install Pyomo in your system Python installation by executing
   the following in a shell:
   ::
        pip install pao

Conditional Dependencies
------------------------

Both **conda** and **pip** can be used to install the third-party packages
that are needed to model problems with PAO.  We recommend **conda**
because it has better support for optimization solver packages.

PAO intrinsically depends on Pyomo, both for the representation of
algebraic problems but also for interfaces to numerical optimizers used by
PAO solvers.  The Pyomo website [PyomoWeb]_ and GitHub site [PyomoGithub]_
provide additional resources for installing Pyomo and related software.

PAO and Pyomo have conditional dependencies on a variety of third-party
packages, including Python packages like scipy, numpy and optimization
solvers.  Optimization solvers are particularly important, and a
commercial optimizer may be needed to analyze complex, real-world
applications.

The following optimizers are the default optimizers used in various PAO solvers:

* `glpk <https://www.gnu.org/software/glpk/>` - An open-source mixed-integer linear programming solver

* `gurobi <https://www.gurobi.com/>` - A commercial mixed-integer linear programming solver that is more robust than glpk

* `ipopt <https://github.com/coin-or/Ipopt>` - An open-source interior point optimizer for continuous problems


