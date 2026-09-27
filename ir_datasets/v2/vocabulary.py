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

A type may declare a ``parent`` (another qualified type, same or different
provider's namespace) -- a single-inheritance hierarchy, not a lattice, so
"is this a Table" stays a plain walk-to-root rather than a graph search. This
is what lets ``irds:QrelTable``/``irds:DocTable``/... each be their own
first-class type (so ``list(type='irds:QrelTable')`` finds exactly qrels
tables) while ``list(type='irds:Table')`` still finds all of them -- and lets
a third-party provider declare ``ext:MyTable`` as a subtype of ``irds:Table``
without ``irds`` knowing ``ext`` exists.
"""

_TYPES = {}   # qualified type -> {'desc': ..., 'owner': prefix, 'parent': qualified | None}
_KINDS = {}   # qualified kind -> {'structural': bool, 'desc': ..., 'owner': prefix}


def declare_type(qualified, *, owner, desc=None, parent=None):
    """Idempotent, like ``declare_kind`` below -- a type may be declared more
    than once (a live import's ``node_type()`` call, then that same
    provider's own manifest reloaded from disk on top of it, or vice versa).
    A re-declaration with ``parent=None`` is treated as "not specified" rather
    than "no parent" -- an older, pre-hierarchy manifest on disk has no
    ``parent`` field at all, and reloading it must not erase a parent the live
    class hierarchy already established. Only a genuine conflict (two
    *different*, both non-``None`` parents) is an error.
    """
    entry = _TYPES.get(qualified)
    if entry is not None:
        if parent is not None:
            if entry.get('parent') not in (None, parent):
                raise ValueError(
                    f'node type {qualified!r} is already declared with parent '
                    f'{entry["parent"]!r}, not {parent!r}')
            entry['parent'] = parent
        return qualified
    # `parent` is stored as given even if it isn't declared *yet* -- a
    # provider that isn't imported may have its own manifest loaded before
    # the provider owning its parent type does; `ancestors()` just stops
    # early until that entry shows up too (same "degrade gracefully for an
    # unknown/not-yet-loaded provider" posture as `is_structural` below).
    _TYPES[qualified] = {'desc': desc, 'owner': owner, 'parent': parent}
    return qualified


def parent_of(qualified):
    entry = _TYPES.get(qualified)
    return entry.get('parent') if entry else None


def ancestors(qualified):
    """``qualified`` and every declared ancestor, closest first."""
    out, seen, cur = [], set(), qualified
    while cur is not None and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = parent_of(cur)
    return out


def is_subtype(sub, of):
    """Whether ``sub`` is ``of`` itself, or a (transitive) subtype of it.

    An undeclared/unknown ``sub`` (e.g. from a provider that isn't installed)
    matches nothing but itself would -- there's no hierarchy to consult.
    """
    if sub is None:
        return False
    return of in ancestors(sub)


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
