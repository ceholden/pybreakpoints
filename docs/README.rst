=============================================
pybreakpoints Documentation
=============================================

Documentation built using `Sphinx <http://sphinx-doc.org/>`__ and hosted
on `Github Pages <https://pages.github.com/>`__.

Building
--------

HTML
~~~~

You can build the HTML files for the documentation using Sphinx.
First, make sure the dependencies for the documentation generation are
installed by ``pip`` installing the ``docs/requirements.txt`` file:

.. code:: bash

    pip install -r requirements.txt

With Sphinx and other packages installed, use the ``Makefile`` to
regenerate the HTML content:

.. code:: bash

    make html

API
~~~

If you have written any code or changed the docstrings in any code, you
will need to update the references to the code in the documentation.
Regenerate API module information for Sphinx

.. code:: bash

    $ sphinx-apidoc -f -e -o source/pybreakpoints ../pybreakpoints/

Guides
~~~~~~

Guide information is written in Restructured Text and involves no
auto-generated information.

Publishing
----------

One way is using `ghp-import <https://github.com/davisp/ghp-import>`__ utility
to push into ``gh-pages`` branch.

.. code:: bash

    [ ceholden@ceholden-llap: yatsm ]$ ghp-import -h
    Usage: ghp-import [OPTIONS] DIRECTORY

    Options:
      -n          Include a .nojekyll file in the branch.
      -m MESG     The commit message to use on the target branch.
      -p          Push the branch to origin/{branch} after committing.
      -r REMOTE   The name of the remote to push to. [origin]
      -b BRANCH   Name of the branch to write to. [gh-pages]
      -h, --help  show this help message and exit

Call:

::

    ghp-import -n -m "$MSG" docs/_build/html
