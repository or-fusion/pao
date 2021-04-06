"""
Script to generate the installer for pao.
"""

import sys
import os


def _find_packages(path):
    """
    Generate a list of nested packages
    """
    pkg_list = []
    if not os.path.exists(path):
        return []
    if not os.path.exists(path+os.sep+"__init__.py"):
        return []
    else:
        pkg_list.append(path)
    for root, dirs, files in os.walk(path, topdown=True):
        if root in pkg_list and "__init__.py" in files:
            for name in dirs:
                if os.path.exists(root+os.sep+name+os.sep+"__init__.py"):
                    pkg_list.append(root+os.sep+name)
    return [pkg for pkg in map(lambda x:x.replace(os.sep, "."), pkg_list)]


def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()

requires = [
    'Pyomo',
    'parameterized',
    'scipy',
    'munch'
    ]

from setuptools import setup
import sys

packages = _find_packages('pao')

setup(name='pao',
      version='1.0.1',
      maintainer='William E. Hart',
      maintainer_email='wehart@sandia.gov',
      url='https://github.com/or-fusion/pao',
      license='BSD',
      platforms=["any"],
      description='PAO: Python Adversarial Optimization',
      long_description=read('README.txt'),
      classifiers=[
        #'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        #'Programming Language :: Python :: Implementation :: Jython',
        #'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules' ],
      packages=packages,
      keywords=['optimization'],
      install_requires=requires,
      python_requires='>=3.7',
      )
