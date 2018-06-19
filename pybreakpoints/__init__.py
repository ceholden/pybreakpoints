""" Structural change detection and other statistics
"""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__package__ = 'pybreakpoints'

__docs__ = 'https://ceholden.github.io/{pkg}/'.format(pkg=__package__)
__docs_version__ = '%s/%s' % (__docs__, __version__)
__repo_url__ = 'https://github.com/ceholden/{pkg}'.format(pkg=__package__)
__repo_issues__ = '/'.join([__repo_url__, 'issues'])


# See: http://docs.python-guide.org/en/latest/writing/logging/
import logging  # noqa
try:  # Python 2.7+
    from logging import NullHandler as _NullHandler
except ImportError:
    class _NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(_NullHandler())
