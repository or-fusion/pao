*DRAFT DOCUMENTATION*

# PAO Overview

PAO is a Python-based package for Adversarial Optimization.  PAO extends the modeling concepts in [Pyomo](https://github.com/Pyomo/pyomo) to enable the expression and solution of multi-level optimization problems. The goal of this package is to provide a general modeling and analysis capability, and application exemplars serve to illustrate PAO's general capabilities.

This package was derived from the capabilities in pyomo.bilevel and pyomo.dualize, which are now deprecated.

Pyomo is available under the BSD License, see the LICENSE.txt file.

### Installation

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pyomo.svg?maxAge=3600)](https://pypi.org/project/pao/) [![PyPI downloads](https://img.shields.io/pypi/dm/pyomo.svg?maxAge=21600)](https://pypistats.org/packages/pao)

    pip install pyomocommunity_pao
    
### Testing

Pyomo is currently tested with the following Python implementations:

* CPython: 3.8

Testing 

* pip install nose coverage

* Simple tests

  * nosetests .

* Tests with coverage

  * nosetests --with-xunit --with-coverage --cover-xml .
  * coverage report -m

### Tutorials and Examples

* TBD

### Getting Help

* [Add a Ticket](https://github.com/pyomocommunity/pao/issues/new)
* [Find a Ticket](https://github.com/pyomocommunity/pao/issues) and **Vote On It**!

### Developers

By contributing to this software project, you are agreeing to the following terms and conditions for your contributions:

1. You agree your contributions are submitted under the BSD license. 
2. You represent you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer.


