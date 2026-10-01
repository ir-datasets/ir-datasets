"""``Node``, ``Edge``, and ``Generator`` -- the graph machinery this package's
own node types (``Resource``, ``Table``, ``Benchmark``, ``Suite``) and any
third-party provider's build on.

This module knows nothing about datasets. It declares no vocabulary of its
own -- no node type, no edge kind -- because that's a provider's job (see
``registry.ManifestProvider.node_type``/``edge_kind``). ``nodes.py`` is where
the dataset vocabulary actually gets declared, on the ``irds`` provider.

A subclass participates in the graph through four optional hooks on ``Node``,
all derived from its own constructor arguments -- nothing is discovered by
inspection:

    structural_edges()   the node's dependencies, as Edges
    attest(...)          what ``freeze`` records about the node (counts,
                         hashes, ...); None for nothing
    verify(frozen)       compare the live node with a frozen attestation
    (metadata)           the descriptive dict written to the manifest row

``citation`` sits alongside ``desc``/``pretty_name`` as a field any node can
carry: "how do I cite this" is not specific to any one node type. It is plain
text here; a richer citation model (papers as nodes, citations as edges) is
future work -- see the README.
"""
import itertools
import re
from typing import NamedTuple, Union

from .context import default
from .vocabulary import is_structural


class Literal:
    """Marks a triple's object as literal data (a string, number, or JSON
    blob) rather than another node's qualified name -- the only thing
    ``discover_edges()``'s ``(subject, predicate, object)`` triples need to
    say to distinguish "this is an edge" from "this is a property value";
    everything else about the triple (which predicates are edge kinds vs.
    field names) is already known from the vocabulary. Named after RDF's own
    "literal" (a value, not a resource reference) -- unrelated to Python's
    ``typing.Literal``, which is a static-typing annotation, not a runtime
    value wrapper; don't confuse the two.
    """
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'Literal({self.value!r})'

    def __eq__(self, other):
        return isinstance(other, Literal) and self.value == other.value

    def __hash__(self):
        return hash(('ir_datasets.v2.Literal', self.value))


class Edge(NamedTuple):
    """A named relationship to another node.

    A node declares its structural edges with *object* targets (intra-package:
    refactor-safe, and the provider registers them along with the node) or
    *name* targets (cross-package: resolved lazily, validated at freeze). In a
    provider and in a manifest, targets are always qualified names -- which is
    why every node having a name matters: the graph is traversable with
    nothing imported.

    Edge *kinds* are declared by providers and qualified like everything else
    (``irds:derived_from``); see ``vocabulary``.
    """
    kind: str
    target: Union[str, 'Node']

    @property
    def structural(self):
        return is_structural(self.kind)

    @property
    def target_name(self):
        t = self.target
        return t if isinstance(t, str) else (t.qualified_name or t.name)

    def __repr__(self):
        return f'{self.kind}->{self.target_name}'


class Node:
    """A name, a ``type`` (set on the subclass, and *declared by a provider*:
    ``type = irds.node_type('Table')``), and a dict of descriptive
    ``metadata``.

    Constructing a node creates an object and nothing else. It joins a graph
    only when a module registers it (``provider.register(...)``), at which
    point the provider stamps ``qualified_name`` (prefix applied),
    ``provider`` and ``defined_in``.
    """
    type = None

    def __init__(self, name, *, desc=None, pretty_name=None, citation=None,
                 deprecated=None, license=None, metadata=None):
        if ':' in name:
            raise ValueError(
                f'{name!r}: node names may not contain ":"; the prefix is '
                f'applied by the provider at registration')
        self.name = name
        deprecated = default('deprecated', deprecated)
        #: The license the node's data is distributed under: an SPDX
        #: identifier (``'CC-BY-4.0'``) where one exists, else the URL of the
        #: governing terms; a list if several apply. Declared in code and
        #: reported at discovery (``discovery_literals``), never frozen into
        #: the manifest -- the declaration *is* the source of truth.
        self.license = default('license', license)
        self.metadata = {k: v for k, v in {
            'desc': desc, 'pretty_name': pretty_name, 'citation': citation,
            'deprecated': deprecated, **(metadata or {}),
        }.items() if v}
        # Assigned by the provider at registration.
        self.qualified_name = None
        self.provider = None
        self.defined_in = None

    def structural_edges(self):
        """This node's dependencies, as Edges (object or name targets)."""
        return []

    def discovery_literals(self):
        """Literal properties (``{field: value}``) reported when the graph is
        discovered, but *not* written to the frozen manifest: facts derivable
        from the node's own declaration (a Resource's size and how it is
        validated), which a manifest row would only duplicate. Empty values are
        dropped."""
        return {'license': self.license} if self.license else {}

    def attest(self, *, verify=False, **options):
        """Data ``freeze`` should record for this node, or None.

        ``verify=True`` permits expensive materialization (iterating a table);
        without it, return only what is cheap. Keys are merged into the node's
        manifest row, so they must not collide with ``name``/``type``/``module``
        or the node's own metadata.
        """
        return None

    def verify(self, frozen, **options):
        """Divergences between this node and its frozen row (``[]`` if none, or
        if there is nothing to compare)."""
        return []

    def _frozen(self):
        """This node's manifest row, if it is registered and frozen."""
        if self.provider is None or self.qualified_name is None:
            return {}
        return self.provider.frozen(self.qualified_name)

    def __repr__(self):
        return f'{type(self).__name__}({(self.qualified_name or self.name)!r})'


# ── Generator (parametric / dynamic families) ────────────────────────────────

class Param:
    def __init__(self, values=None, pattern=None, desc=None, default=None):
        self.values = tuple(values) if values else None
        self.pattern = pattern
        self.desc = desc
        self.default = default

    def regex(self):
        if self.values:
            return '(?:' + '|'.join(re.escape(v) for v in self.values) + ')'
        return self.pattern or r'[^-]+'


class Generator:
    """A rule that produces nodes on demand.

    Used where enumeration is impractical (a 139x139x4x3 parametric family) or
    impossible (a family defined on a remote server). ``freeze`` records the
    rule -- template, parameter domains, metadata -- not its expansion.
    ``type`` is the (qualified, declared) node type of what it produces.

    The name template doubles as the parameter spec, so name<->params
    round-trips by construction rather than via a regex kept in sync by hand.
    """

    def __init__(self, template, *, params, type, resolver, constraints=(),
                 enumerable=True, edges=(), row_metadata=None,
                 defined_in=None, **meta):
        self.template = template
        self.params = params
        self.node_type = type
        #: Called with the matched parameters to build the node. Usually the
        #: node class itself, whose constructor takes the same parameter names
        #: as the template.
        self.resolver = resolver
        self.constraints = list(constraints)
        self.enumerable = enumerable
        #: [(kind, target_template), ...] -- this generator's products'
        #: structural edges, declared the same way `template` names the
        #: product itself: a format string over the same params, not a real
        #: lookup. Lets `discover_edges()` report a product's dependencies
        #: (e.g. a Benchmark generator's `docs`/`queries`/`qrels` facets)
        #: for every enumerated name without resolving it -- descriptive
        #: metadata only, like `type`; never checked against what
        #: `resolve_node()` actually builds.
        self.edges = list(edges)
        #: Optional ``**params -> dict``, called once per enumerated name
        #: (see ``enumerate_rows``) to report cheap per-name properties --
        #: e.g. a Resource generator's ``sources``/``hashes`` -- without
        #: running the full ``resolver`` (which may build a parser, wrap a
        #: docstore, ...). "Cheap" is the caller's responsibility: this is
        #: still invoked once per enumerated name, so it should amortize any
        #: real work (a remote index fetch, say) behind its own cache rather
        #: than repeating it -- unlike `edges=`, which is pure string
        #: substitution and therefore always free.
        self.row_metadata = row_metadata
        self.meta = meta
        # Assigned by provider.register_generator.
        self.provider = None
        self.defined_in = defined_in
        pattern = re.escape(template)
        for key, param in params.items():
            pattern = pattern.replace(re.escape('{' + key + '}'),
                                      f'(?P<{key}>{param.regex()})')
        self._regex = re.compile(f'^{pattern}$')

    def format(self, **params):
        return self.template.format(**params)

    def match(self, name):
        prefix = self.provider.prefix if self.provider is not None else ''
        if prefix:
            if not name.startswith(f'{prefix}:'):
                return None
            name = name[len(prefix) + 1:]
        m = self._regex.match(name)
        if m is None:
            return None
        params = m.groupdict()
        if not all(c(**params) for c in self.constraints):
            return None
        return params

    def validate(self, **params):
        for key, param in self.params.items():
            if param.values and params.get(key) not in param.values:
                raise ValueError(
                    f'{key}={params.get(key)!r} is not valid; expected one of '
                    f'{param.values[:8]}{"..." if len(param.values) > 8 else ""}')
        for constraint in self.constraints:
            if not constraint(**params):
                raise ValueError(f'invalid parameter combination: {params}')

    def resolve_node(self, params):
        self.validate(**params)
        return self.resolver(**params)

    def enumerate_params(self):
        """Every valid params dict this generator's closed value sets
        produce -- the cross product of every param's ``values``,
        constraint-filtered. Only meaningful when every param has one (an
        open-ended ``pattern=`` param, like ``hf``'s, has no finite cross
        product -- that's exactly why ``hf`` declares ``enumerable=False``);
        a caller is expected to only ever call this when ``enumerable`` is
        true, so a mismatch here is a real bug, not a routine case to
        degrade gracefully from."""
        missing = [k for k, p in self.params.items() if not p.values]
        if missing:
            raise ValueError(
                f'{self!r} cannot be enumerated: {", ".join(missing)} has no '
                f'closed values= set')
        keys = list(self.params)
        for combo in itertools.product(*(self.params[k].values for k in keys)):
            params = dict(zip(keys, combo))
            if all(c(**params) for c in self.constraints):
                yield params

    def _qualify(self, name):
        return self.provider.qualify(name) if self.provider else name

    def enumerate(self):
        """Every (qualified) name this generator can produce -- see
        ``enumerate_params``."""
        for params in self.enumerate_params():
            yield self._qualify(self.template.format(**params))

    def enumerate_edges(self):
        """``(name, kind, target)`` for every name this generator can produce
        and every edge declared in ``edges=`` -- both this generator's own
        template and each edge's target template are formatted with the same
        params, so a Benchmark generator's
        ``edges=[('irds:docs', 'family-{lang}-docs')]`` reports the real
        target name without resolving anything. ``target`` is always another
        node's (qualified) name, never wrapped in ``Literal`` -- matching the
        shape ``discover_edges()`` uses everywhere else."""
        for params in self.enumerate_params():
            name = self._qualify(self.template.format(**params))
            for kind, target_template in self.edges:
                yield (name, self._qualify(kind),
                      self._qualify(target_template.format(**params)))

    def enumerate_rows(self):
        """(name, row) for every name this generator can produce, ``row``
        being ``{'type': ..., **row_metadata(**params)}`` -- only meaningful
        when ``row_metadata=`` is set (a caller checks that first; this
        doesn't degrade to bare-type rows on its own, since a caller wanting
        those already has the cheaper ``enumerate()``)."""
        for params in self.enumerate_params():
            name = self._qualify(self.template.format(**params))
            yield name, {'type': self.node_type, **self.row_metadata(**params)}

    def metadata(self):
        resolver = getattr(self.resolver, '__qualname__', repr(self.resolver))
        return {'name_template': self.template, 'kind': 'generator',
                'type': self.node_type, 'module': self.defined_in,
                'resolver': resolver, 'enumerable': self.enumerable,
                'params': {k: {'values': list(v.values) if v.values else None,
                               'pattern': v.pattern, 'desc': v.desc}
                           for k, v in self.params.items()},
                'edges': [{'kind': k, 'target_template': t} for k, t in self.edges],
                **self.meta}

    def __repr__(self):
        return f'Generator({self.template!r})'
