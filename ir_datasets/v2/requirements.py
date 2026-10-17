"""Optional dependencies and where a node comes from.

The core install is deliberately small (``requests``, ``lz4``, ``numpy``,
``tqdm``, ``pyyaml``, ``pyarrow``). A node that needs anything else names the
**pip packages** it needs::

    DocTable('dpr-w100-docs', ..., requires=['ijson'])

or, for a table whose *format* is what needs them, on its parser class::

    class _DprW100QrelsParser(Parser):
        requires = ('ijson',)

Names are distribution names as on PyPI (``trec-car-tools``, not an import
name and not an ``ir_datasets`` extra). ``Graph.load`` warns when a node (or
anything it is built from) needs a package that isn't installed, and a node's
own ``requires`` is reported at discovery (as ``requires-dependency``), along
with the ``package`` it comes from (see ``distribution_of``).
"""
import importlib.metadata as metadata
import re
import warnings

_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


class MissingDependencyWarning(UserWarning):
    """A node needs a package that isn't installed."""


def canonical(name):
    """PEP 503 normalized form, for comparing names (``Huggingface_Hub`` ==
    ``huggingface-hub``)."""
    return re.sub(r'[-_.]+', '-', name).lower()


def normalize(requires):
    """``requires`` as a de-duplicated tuple of package names, in the order given."""
    out = {}
    for name in requires or ():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f'{name!r} is not a package name (give a pip distribution '
                             f'name like "trec-car-tools", without a version or extra)')
        out.setdefault(canonical(name), name)
    return tuple(out.values())


def installed_version(name):
    """The installed version of a distribution, or ``None`` if it isn't."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def is_installed(name):
    return installed_version(name) is not None


def missing(requires):
    """The subset of ``requires`` that isn't installed."""
    return [n for n in normalize(requires) if not is_installed(n)]


def install_hint(requires):
    return f'pip install {" ".join(normalize(requires))}'


def warn_if_missing(name, requires, stacklevel=3):
    absent = missing(requires)
    if absent:
        warnings.warn(
            f'{name} needs packages that are not installed ({", ".join(absent)}); '
            f'using it will fail. Install them with {install_hint(absent)}',
            MissingDependencyWarning, stacklevel=stacklevel)
    return absent


_top_level = None


def distribution_of(module):
    """The name of the installed distribution that provides the top-level
    package of ``module`` (``'ir_datasets.v2.nodes'`` -> ``'ir_datasets'``), or
    ``None`` if it can't be told (a script, a namespace package, an uninstalled
    checkout)."""
    global _top_level
    if not module:
        return None
    if _top_level is None:
        _top_level = metadata.packages_distributions()
    names = _top_level.get(module.split('.')[0])
    return names[0] if names else None
