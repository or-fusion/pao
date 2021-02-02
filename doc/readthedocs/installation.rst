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

PAO and Pyomo have conditional dependencies on a variety of third-party
packages, including Python packages like scipy, numpy and optimization
solvers.  Both **conda** and **pip** install the third-party packages that
are needed to model problems with PAO.  However, a PAO user may need to
install optimization solvers to analysis complex, real-world applications.

PAO intrinsically depends on Pyomo, both for the representation of
algebraic problems but also for interfaces to numerical optimizers used by
PAO solvers.  The Pyomo website [PyomoWeb]_ and GitHub site [PyomoGithub]_
provide additional resources for installing Pyomo and related software.

