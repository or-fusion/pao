Installation
============

PAO currently supports the following versions of Python:

* CPython: 3.7, 3.8, 3.9


Using GIT
---------

PAO can be installed by cloning the PAO software repostory and 
then directly installing the software.  For example, the master
branch can be installed as follows:

::
   
   git clone https://github.com/or-fusion/pao.git
   cd pao
   python setup.py develop

Using PIP
---------

The standard utility for installing Python packages is **pip**.  
You can install the latest release of PAO by executing the following:

::

    pip install pao

You can also use **pip** to install from the PAO software repository.
For example, the master branch can be installed as follows:

::

    python -m pip install https://github.com/or-fusion/pao.git

.. note::

    Support for Conda installation is planned.


Conditional Dependencies
------------------------

Both **conda** and **pip** can be used to install the third-party packages
that are needed to model problems with PAO.  We recommend **conda**
because it has better support for optimization solver packages.

PAO intrinsically depends on `Pyomo <https://github.com/Pyomo/pyomo>`_,
both for the representation of algebraic problems but also for
interfaces to numerical optimizers used by PAO solvers.  `Pyomo
<https://github.com/Pyomo/pyomo>`_ is installed with PAO, but the Pyomo
website [PyomoWeb]_ and GitHub site [PyomoGithub]_ provide additional
resources for installing Pyomo and related software.

PAO and `Pyomo <https://github.com/Pyomo/pyomo>`_ have conditional
dependencies on a variety of third-party packages, including Python
packages like scipy, numpy and optimization solvers.  Optimization solvers
are particularly important, and a commercial optimizer may be needed to
analyze complex, real-world applications.

The following optimizers are used to test the PAO solvers:

* `glpk <https://www.gnu.org/software/glpk/>`_ - An open-source mixed-integer linear programming solver

* `cbc <https://github.com/coin-or/Cbc>`_ - An open-source mixed-integer linear programming solver

* `ipopt <https://github.com/coin-or/Ipopt>`_ - An open-source interior point optimizer for continuous problems

Additionally, PAO can interface with optimization solvers at `NEOS <https://neos-server.org/neos/>`_.

