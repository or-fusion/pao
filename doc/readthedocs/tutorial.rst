Tutorial
========

The following sections provide a tutorial for PAO, focusing on the types
of solvers supported in PAO and their use.

Consider the following simple workflow supported by PAO:

1. Create a multi-level model, represented with Pyomo or LinearMultilevelProblem objects

2. Create a solver using the Solver functor

3. Apply the solver to the problem

4. Get the final solution from the model object

The Solver is a global object that is used to create a solver.
This simplifies a user's workflow by eliminating the need to 
import the Python modules containing the PAO solvers.  Instead, a user
can give the name of the solver as a string.

The following list summarizes the solvers currently supported in PAO:

* FA

  * pao.pyomo.FA, pao.mpr.FA

  * Linear bilevel problems with a continuous subproblem.

  * Uses reformulation of subproblem using KKT conditions, solved with a MIP solver.

* REG

  * pao.pyomo.REG, pao.mpr.REG

  * Linear bilevel problems with a continuous subproblem.  

  * Uses reformulation of subproblem using KKT conditions, solved with a NLP solver.

* PCCG

  * pao.pyomo.PCCG, pao.mpr.PCCG

  * Linear bilevel problems with a continuous or integer subproblem.

  * Uses a projection-based reformulation and decomposition algorithm, where subproblems are solved with a MIP solver.
    
The following section illustrate these solvers on illustrative examples.

FA Solver
---------
TODO

REG Solver
----------
TODO

PCCG Solver
-----------
