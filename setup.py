#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa E501
"""A setuptools based setup module for pybreakpoints
"""
from codecs import open
from os import path
from setuptools import setup, find_packages
import sys

import versioneer
version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

PY2 = sys.version_info[0] == 2
HERE = path.abspath(path.dirname(__file__))
DOCS = path.join(HERE, 'docs', 'source')

README = path.join(HERE, 'README.rst')
HISTORY = path.join(DOCS, 'history.rst')

with open(README, encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(HISTORY, encoding='utf-8') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requires = {
    'core': ['six', 'numpy', 'pandas', 'scipy', 'xarray'],
    'docs': [
        'mock', 'sphinx>=1.4', 'sphinx_rtd_theme', 'sphinxcontrib-bibtex',
        'sphinx-paramlinks'
    ],
    'test': ['pytest', 'coverage', 'mock'],
    'numba': ['numba']
}

if PY2:
    requires['core'].extend([
        'pathlib',
    ])

requires['all'] = sorted(set(sum(requires.values(), [])))

packages = sorted(find_packages(exclude=['docs', 'tests']))
entry_points = {}

setup(
    name='pybreakpoints',
    version=version,
    cmdclass=cmdclass,
    description="Break point detection algorithms in Python",
    long_description=readme + '\n\n' + history,
    author="Chris Holden",
    author_email='ceholden@gmail.com',
    url='https://github.com/ceholden/pybreakpoints',
    packages=packages,
    entry_points=entry_points,
    include_package_data=True,
    install_requires=requires['core'],
    license="BSD-3",
    classifiers=[],
    tests_require=requires['test']
)
