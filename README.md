# pao

A Python Package for Adversarial Optimization

# Notes

* This package contains Python software for adversarial optimization that
is not application-specific.  The goal of this package is to support
cross-cutting capabilities that can be leveraged by multiple projects.

* This package has been initialized with current capabilities in pyomo.bilevel and pyomo.dualized.


# Testing

* pip install nose coverage

* Simple tests

  * nosetests .
  * This recursively runs an summarizes tests in the test\*.py files

* Tests with coverage

  * nosetests --with-xunit --with-coverage --cover-xml .
  * coverage report -m
