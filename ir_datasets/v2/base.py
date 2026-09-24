"""``Node``, ``Edge``, and ``Generator`` -- the graph machinery this package's
own node types (``Resource``, ``Table``, ``Benchmark``, ``Suite``) and any
third-party provider's build on.

This module knows nothing about datasets. It declares no vocabulary of its
own -- no node type, no edge kind -- because that's a provider's job (see
``registry.Provider.node_type``/``edge_kind``). ``nodes.py`` is where the
dataset vocabulary actually gets declared, on the ``irds`` provider.

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
import re
from typing import NamedTuple, Union

from .context import default
from .vocabulary import is_structural


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
                 deprecated=None, metadata=None):
        if ':' in name:
            raise ValueError(
                f'{name!r}: node names may not contain ":"; the prefix is '
                f'applied by the provider at registration')
        self.name = name
        deprecated = default('deprecated', deprecated)
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
                 enumerable=True, defined_in=None, **meta):
        self.template = template
        self.params = params
        self.node_type = type
        #: Called with the matched parameters to build the node. Usually the
        #: node class itself, whose constructor takes the same parameter names
        #: as the template.
        self.resolver = resolver
        self.constraints = list(constraints)
        self.enumerable = enumerable
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

    def metadata(self):
        resolver = getattr(self.resolver, '__qualname__', repr(self.resolver))
        return {'name_template': self.template, 'kind': 'generator',
                'type': self.node_type, 'module': self.defined_in,
                'resolver': resolver, 'enumerable': self.enumerable,
                'params': {k: {'values': list(v.values) if v.values else None,
                               'pattern': v.pattern, 'desc': v.desc}
                           for k, v in self.params.items()},
                **self.meta}

    def __repr__(self):
        return f'Generator({self.template!r})'
