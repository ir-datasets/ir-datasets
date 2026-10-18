"""The process-wide index of declared graph classes and graph properties.

Nothing is declared *here*. A graph class or graph property is declared by a
provider (``provider.declare_graph_class('Table')``,
``provider.declare_graph_property('derivedFrom', structural=True)``,
``provider.declare_graph_property('description', range='literal')``) and carries that
provider's prefix: ``irds:Table``, ``irds:derivedFrom``, ``irds:description``.
Graph classes are UpperCamelCase and graph properties lowerCamelCase, as RDF/OWL
vocabularies write them. A graph property is either an edge between nodes or a
literal-valued property (a value or a JSON document); every triple in the graph
uses a declared one. This module is only the process-wide index
``registry.py`` consults to validate a node's type at registration, and to
tell structural kinds from informational ones when traversing. A provider's
vocabulary is known once its modules have been imported.

A type may declare a ``parent`` (another qualified type, same or different
provider's namespace) -- a single-inheritance hierarchy, not a lattice, so
"is this a Table" stays a plain walk-to-root rather than a graph search. This
is what lets ``irds:Qrels``/``irds:Docs``/... each be their own
first-class type (so ``list(type='irds:Qrels')`` finds exactly qrels
tables) while ``list(type='irds:Table')`` still finds all of them -- and lets
a third-party provider declare ``ext:MyTable`` as a subtype of ``irds:Table``
without ``irds`` knowing ``ext`` exists.
"""

_CLASSES = {}   # qualified type -> {'desc', 'owner': prefix, 'parent': qualified | None, 'protocol': 'module:Name' | None}
_PROPERTIES = {}   # qualified graph property -> {'range', 'structural', 'desc', 'owner': prefix}

#: The standard RDF graph properties for a node's type and the type hierarchy. Built in:
#: declared here rather than by a provider (they are not ours to own).
RDF_TYPE = 'rdf:type'
RDFS_SUBCLASS_OF = 'rdfs:subClassOf'

#: A graph property's range: ``node`` (an edge to another node), ``literal`` (a plain
#: value) or ``json`` (a literal holding a JSON document), or ``type`` (a node
#: type, for the two built-ins).
RANGES = ('node', 'literal', 'json', 'type')


def declare_graph_class(qualified, *, owner, desc=None, parent=None, protocol=None):
    """Idempotent, like ``declare_kind`` below -- a type may be declared more
    than once. A re-declaration with ``parent=None`` is treated as "not
    specified" rather than "no parent", so it cannot erase a parent the live
    class hierarchy already established. Only a genuine conflict (two
    *different*, both non-``None`` parents) is an error.
    """
    entry = _CLASSES.get(qualified)
    if entry is not None:
        if parent is not None:
            if entry.get('parent') not in (None, parent):
                raise ValueError(
                    f'graph class {qualified!r} is already declared with parent '
                    f'{entry["parent"]!r}, not {parent!r}')
            entry['parent'] = parent
        # Same "None means not specified" rule as `parent`.
        if protocol is not None:
            entry['protocol'] = protocol
        return qualified
    # `parent` is stored as given even if it isn't declared *yet* -- the
    # provider owning it may not have been imported; `ancestors()` just stops
    # early until that entry shows up (same "degrade gracefully for an
    # unknown/not-yet-loaded provider" posture as `is_structural` below).
    _CLASSES[qualified] = {'desc': desc, 'owner': owner, 'parent': parent, 'protocol': protocol}
    return qualified


def parent_of(qualified):
    entry = _CLASSES.get(qualified)
    return entry.get('parent') if entry else None


def protocol_of(qualified):
    """``(protocol, declared_on)`` for a type: the ``'module:Name'`` protocol
    reference declared on it or, failing that, its closest ancestor that has
    one -- and which type that was. ``(None, None)`` if none do."""
    for t in ancestors(qualified):
        entry = _CLASSES.get(t)
        if entry and entry.get('protocol'):
            return entry['protocol'], t
    return None, None


def type_for_protocol(ref):
    """The qualified type whose declared protocol is ``ref`` (``'module:Name'``),
    or ``None``. Used to find the parent type of a protocol that extends another."""
    for qualified, entry in _CLASSES.items():
        if entry.get('protocol') == ref:
            return qualified
    return None


def ancestors(qualified):
    """``qualified`` and every declared ancestor, closest first."""
    out, seen, cur = [], set(), qualified
    while cur is not None and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = parent_of(cur)
    return out


def is_subclass(sub, of):
    """Whether ``sub`` is ``of`` itself, or a (transitive) subtype of it.

    An undeclared/unknown ``sub`` (e.g. from a provider that isn't installed)
    matches nothing but itself would -- there's no hierarchy to consult.
    """
    if sub is None:
        return False
    return of in ancestors(sub)


def types_of(node):
    """Every graph class of a node: ``node.types`` if it has one (a node may be
    several things at once, e.g. a benchmark that is both ad hoc and QA),
    else just ``(node.type,)``. The first entry is always ``node.type``."""
    return tuple(getattr(node, 'types', None) or (node.type,))


_KEY_ORDER = ("type", "name", "order", "data_type")


def canonical_json(value):
    """``value`` as JSON text with a stable layout: in every object the keys in
    ``_KEY_ORDER`` (``type``; a column's ``name``, ``order``, ``data_type``) come
    first in that order, then the others sorted -- so a JSON-valued property
    reads the same way each time it is rendered."""
    import json

    def ordered(v):
        if isinstance(v, dict):
            head = {k: ordered(v[k]) for k in _KEY_ORDER if k in v}
            return {**head, **{k: ordered(v[k]) for k in sorted(v) if k not in head}}
        if isinstance(v, (list, tuple)):
            return [ordered(x) for x in v]
        return v
    return json.dumps(ordered(value))


def row_for(node):
    """A node's descriptive row: name, type(s), defining module, the
    ``package`` it comes from, and metadata. Empty values are dropped so rows
    stay readable."""
    from .requirements import distribution_of
    types = types_of(node)
    package = distribution_of(node.defined_in) or getattr(node.provider, 'distribution', None)
    row = {'name': node.qualified_name, 'type': types[0] if len(types) == 1 else list(types),
           'module': node.defined_in, 'package': package, **node.metadata}
    return {k: v for k, v in row.items() if v or k == 'name'}


def declare_graph_property(qualified, *, owner, range='node', structural=False, desc=None):
    """Idempotent. A graph property is either an edge to another node (``range='node'``,
    structural or informational) or a property: a ``literal`` value or a ``json``
    document. Like any RDF property it may have several values on a node."""
    if range not in RANGES:
        raise ValueError(f'graph property {qualified!r}: range must be one of {RANGES}, not {range!r}')
    if structural and range != 'node':
        raise ValueError(f'graph property {qualified!r}: only an edge (range="node") can be structural')
    entry = _PROPERTIES.get(qualified)
    if entry is not None:
        if (entry['range'], entry['structural']) != (range, structural):
            raise ValueError(f'graph property {qualified!r} is already declared differently '
                             f'(range={entry["range"]!r}, structural={entry["structural"]})')
        return qualified
    _PROPERTIES[qualified] = {'range': range, 'structural': structural,
                              'desc': desc, 'owner': owner}
    return qualified


declare_graph_property(RDF_TYPE, owner='rdf', range='type', desc="a node's type")
declare_graph_property(RDFS_SUBCLASS_OF, owner='rdfs', range='type', desc='a type -> its parent type')


def graph_properties():
    """Every declared graph property: edges, properties and the built-in RDF ones."""
    return {k: dict(v) for k, v in _PROPERTIES.items()}


def property_for(field, prefix):
    """The declared property (a literal/json graph property) a row field stands for:
    ``<prefix>:<field>`` if the node's own provider declares one, else any other
    provider's of that name; ``None`` if none is declared."""
    own = f'{prefix}:{field}'
    entry = _PROPERTIES.get(own)
    if entry is not None and entry['range'] in ('literal', 'json'):
        return own
    for qualified, entry in sorted(_PROPERTIES.items()):
        if entry['range'] in ('literal', 'json') and qualified.split(':', 1)[1] == field:
            return qualified
    return None


def graph_classes():
    return {k: dict(v) for k, v in _CLASSES.items()}


def edge_kinds():
    """The declared graph properties that are edges between nodes."""
    return {k: dict(v) for k, v in _PROPERTIES.items() if v['range'] == 'node'}


def literal_properties():
    """The declared graph properties that carry a literal or JSON value."""
    return {k: dict(v) for k, v in _PROPERTIES.items() if v['range'] in ('literal', 'json')}


def structural_kinds():
    return frozenset(k for k, v in _PROPERTIES.items() if v['structural'])


def informational_kinds():
    return frozenset(k for k, v in _PROPERTIES.items() if v['range'] == 'node' and not v['structural'])


def is_structural(kind):
    """Unknown kinds are informational, so that traversing a graph that mentions
    a package that is not installed degrades to "not part of the build"
    rather than failing."""
    entry = _PROPERTIES.get(kind)
    return bool(entry and entry['structural'])


def check_graph_class(type):
    if type not in _CLASSES:
        raise ValueError(
            f'undeclared graph class {type!r}; a provider must declare it with '
            f'provider.declare_graph_class(...) and the class set type = that value')
    return type


def check_kind(kind):
    entry = _PROPERTIES.get(kind)
    if entry is None or entry['range'] != 'node':
        raise ValueError(
            f'undeclared edge kind {kind!r}; a provider must declare it with '
            f'provider.declare_graph_property(..., structural=...)')
    return kind


class UndeclaredPropertyWarning(UserWarning):
    """A node has a field that no provider has declared as a graph property; it is
    left out of the graph."""


_warned_fields = set()


def warn_undeclared(field, node_name):
    if field not in _warned_fields:
        _warned_fields.add(field)
        import warnings
        warnings.warn(
            f'{node_name} has a field {field!r} that no provider has declared with '
            f'provider.declare_graph_property({field!r}, range=...); it is left out of the graph',
            UndeclaredPropertyWarning, stacklevel=3)
