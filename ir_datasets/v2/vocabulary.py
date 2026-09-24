"""The process-wide index of declared node types and edge kinds.

Nothing is declared *here*. A type or kind is declared by a provider
(``provider.node_type('Table')``, ``provider.edge_kind('derived_from', structural=True)``)
and carries that provider's prefix: ``irds:Table``, ``irds:derived_from``. Types
are capitalized (RDF/OWL convention: classes are UpperCamelCase); kinds stay
lowercase (properties aren't). This module is only the process-wide index
``registry.py`` consults to validate a node's type at registration, and to
tell structural kinds from informational ones when traversing -- including
kinds owned by a provider that is installed but not imported, which the index
learns from that provider's manifest.
"""

_TYPES = {}   # qualified type -> {'desc': ..., 'owner': prefix}
_KINDS = {}   # qualified kind -> {'structural': bool, 'desc': ..., 'owner': prefix}


def declare_type(qualified, *, owner, desc=None):
    entry = _TYPES.get(qualified)
    if entry is None:
        _TYPES[qualified] = {'desc': desc, 'owner': owner}
    return qualified


def declare_kind(qualified, *, structural, owner, desc=None):
    entry = _KINDS.get(qualified)
    if entry is not None:
        if entry['structural'] != structural:
            raise ValueError(
                f'edge kind {qualified!r} is already declared as '
                f'{"structural" if entry["structural"] else "informational"}')
        return qualified
    _KINDS[qualified] = {'structural': structural, 'desc': desc, 'owner': owner}
    return qualified


def node_types():
    return {k: dict(v) for k, v in _TYPES.items()}


def edge_kinds():
    return {k: dict(v) for k, v in _KINDS.items()}


def structural_kinds():
    return frozenset(k for k, v in _KINDS.items() if v['structural'])


def informational_kinds():
    return frozenset(k for k, v in _KINDS.items() if not v['structural'])


def is_structural(kind):
    """Unknown kinds are informational, so that traversing a manifest written
    by a package that is not installed degrades to "not part of the build"
    rather than failing."""
    entry = _KINDS.get(kind)
    return bool(entry and entry['structural'])


def check_type(type):
    if type not in _TYPES:
        raise ValueError(
            f'undeclared node type {type!r}; a provider must declare it with '
            f'provider.node_type(...) and the class set type = that value')
    return type


def check_kind(kind):
    if kind not in _KINDS:
        raise ValueError(
            f'undeclared edge kind {kind!r}; a provider must declare it with '
            f'provider.edge_kind(..., structural=...)')
    return kind
