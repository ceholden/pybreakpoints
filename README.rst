===============================
pybreakpoints
===============================


+---------+---------------------------------------------------------------+
| License |                         |license|                             |
+---------+-----------------+----------------------+----------------------+
| Branch  |          CI     |       Coverage       |       Docs           |
+=========+=================+======================+======================+
| master  | |master_travis| |  |master_coverage|   | |master_docs_ghpage| |
+---------+-----------------+----------------------+----------------------+

Break point detection algorithms in Python (with Numba if installed).


TODO
----

Things need doing.

- Documentation
  + Overview and citations
  + Install guide
  + Algorithms page with examples
  + Other modules
- Fixup return types
  + ``namedtuple``? class?
- Clean wrapping of Pandas and Xarray types
  + likely using ``singledispatch``
- Tests on randomized data with ``rpy2``
  + Use ``hypothesis``?
- Simple benchmarks to make sure Numba is worth it


.. |license| image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
   :target: https://raw.githubusercontent.com/ceholden/pybreakpoints/master/LICENSE

.. |master_travis| image:: https://api.travis-ci.com/ceholden/pybreakpoints.svg?token=fpEUL8V3obFi2DonCumW&branch=master
   :target: https://travis-ci.com/ceholden/pybreakpoints
.. |master_coverage| image:: https://ceholden.github.io/pybreakpoints/master/coverage_badge.svg
    :target: https://ceholden.github.io/pybreakpoints/master/coverage
.. |master_docs_ghpage| image:: https://api.travis-ci.com/ceholden/pybreakpoints.svg?token=fpEUL8V3obFi2DonCumW&branch=master
   :target: https://ceholden.github.io/pybreakpoints/master/
