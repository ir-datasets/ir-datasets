"""``Graph``: the knowledge graph assembled from every installed provider.

Nothing registers *into* a Graph, and a Graph asks nothing of a provider
beyond ``protocols.Provider``: ``load(name)`` to resolve one node, and
``discover_edges()`` to get its whole catalog as RDF-ready
``(subject, predicate, object)`` triples -- ``object`` wrapped in
``base.Literal`` when it's a property value rather than another node's name
(see ``registry.row_triples``). Everything below -- traversal, listing,
export, validation -- is built from those two calls, unchanged. A provider's
catalog is cached the first time a ``Graph`` reads it, since
``discover_edges()`` always does full discovery and may be a live crawl (see
``hf``) -- so a query doesn't repeat that work, but only for as long as this
particular ``Graph`` lives.

``default_graph()`` is the one over every installed entry point in the
``ir_datasets.providers`` group. This package declares no providers of its own
beyond ``irds`` (added explicitly in ``ir_datasets/v2/__init__.py``). A package
may also add its provider directly when it is imported (``default_graph().add``),
which is what makes an editable checkout work before its entry points exist.

There is no default provider -- with one exception, purely for ``load()``'s
own convenience: a bare name (no ``prefix:``) tries ``irds:`` first, then
``legacy:``, since those are the two places an unqualified name is actually
expected -- a plain v2 dataset name, or an old v1 id. Every *other* name still
needs its ``prefix:``.

That ``legacy:`` fallback is also where a v1 id actually gets resolved to its
v2 replacement. ``legacy_provider.py``'s ``legacy:`` provider makes every v1
id a real, browsable ``legacy:V1Dataset`` node, with an ``irds:aliasOf``
edge to its v2 counterpart where one is known -- an ordinary node and edge,
nothing ``Graph`` needs to special-case to list or traverse. Loading one is
the one deliberate exception: whenever ``Graph.__getitem__`` resolves *any*
name (bare or qualified) to a ``irds:Alias`` (``legacy:V1Dataset`` is one), it follows that node's
``alias_of`` edge and returns the v2 node instead -- or raises, if the v1
id has no known replacement, since a ``V1Dataset`` placeholder is browsable
but never itself a working dataset.
"""
import sys

from . import requirements
from .base import Literal
from .vocabulary import (
    RDF_PROPERTY, RDF_TYPE, RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBCLASS_OF, RDFS_SUBPROPERTY_OF, is_subclass,
    graph_properties, structural_kinds,
)

ENTRY_POINT_GROUP = 'ir_datasets.providers'

#: The one graph class ``Graph.__getitem__`` treats specially (see the module
#: docstring) -- not part of the generic ``Provider``/vocabulary contract,
#: just the two things needed to resolve a legacy id to its replacement.
_ALIAS_TYPE = 'irds:Alias'
_ALIAS_OF_KIND = 'irds:aliasOf'


class Graph:
    def __init__(self, providers=()):
        self.providers = {}     # prefix -> Provider
        self._catalogs = {}     # prefix -> catalog dict, from discover_edges()
        for provider in providers:
            self.add(provider)

    def add(self, provider):
        existing = self.providers.get(provider.prefix)
        if existing is not None and existing is not provider:
            raise ValueError(
                f'two providers claim the prefix {provider.prefix!r}: '
                f'{existing} and {provider}')
        self.providers[provider.prefix] = provider
        return provider

    def __repr__(self):
        return f'Graph({sorted(self.providers)})'

    # -- catalog --------------------------------------------------------------

    def _catalog(self, provider):
        """``{'types', 'fwd', 'rev'}`` for one provider, built once from
        ``discover_edges()`` and cached -- everything else in this class
        reads from here rather than calling the provider again."""
        cached = self._catalogs.get(provider.prefix)
        if cached is None:
            types, all_types, fwd, rev = {}, {}, {}, {}
            seen_edges = set()
            for subject, kind, obj in provider.discover_edges():
                if isinstance(obj, Literal):
                    continue
                if kind == RDF_TYPE and obj == RDF_PROPERTY:
                    # A graph property describing itself (see ManifestProvider.property_triples):
                    # vocabulary, not a node.
                    continue
                if kind == RDF_TYPE:
                    # A node may have several types; `types` keeps the first
                    # (its primary), `all_types` every one.
                    types.setdefault(subject, obj)
                    all_types.setdefault(subject, []).append(obj)
                elif kind in (RDFS_SUBCLASS_OF, RDFS_RANGE, RDFS_DOMAIN, RDFS_SUBPROPERTY_OF):
                    # Type-hierarchy metadata (vocabulary.py's own
                    # declare_graph_class/ancestors already track this for
                    # is_subclass) -- not a node-to-node edge, so it must not
                    # show up in edges_of/check's dangling-target search.
                    continue
                else:
                    bucket = fwd.setdefault(subject, {}).setdefault(kind, [])
                    if (subject, kind, obj) not in seen_edges:        # O(1), not a scan of the bucket
                        seen_edges.add((subject, kind, obj))
                        bucket.append(obj)
                    rev.setdefault(obj, set()).add((subject, kind))
            cached = {'types': types, 'all_types': all_types, 'fwd': fwd, 'rev': rev}
            self._catalogs[provider.prefix] = cached
        return cached

    # -- names --------------------------------------------------------------

    def provider_for(self, name):
        """The provider whose namespace a qualified name falls in."""
        if ':' not in name:
            raise KeyError(
                f'{name!r} is not a qualified name; every node lives in a '
                f'provider (<prefix>:{name}; installed prefixes: '
                f'{", ".join(sorted(self.providers)) or "none"})')
        prefix = name.split(':', 1)[0]
        try:
            return self.providers[prefix]
        except KeyError:
            raise KeyError(
                f'no provider for prefix {prefix!r} (needed by {name!r}); is '
                f'the package that provides it installed?') from None

    def _load_bare(self, name):
        """A bare name (no ``prefix:``) tries ``irds:`` then ``legacy:`` --
        see the module docstring -- and fails otherwise. Returns
        ``(node, qualified_name)`` rather than just the node, so the caller
        can still run the ``legacy:V1Dataset`` check below against the right
        name."""
        for prefix in ('irds', 'legacy'):
            provider = self.providers.get(prefix)
            if provider is None:
                continue
            candidate = f'{prefix}:{name}'
            try:
                return provider.load(candidate), candidate
            except KeyError:
                continue
        raise KeyError(
            f'{name!r} is not a qualified name; tried irds:{name} and '
            f'legacy:{name}; every node lives in a provider (<prefix>:{name})')

    def __getitem__(self, name):
        if name.endswith(']') and '[' in name:
            # A facet-swapped benchmark spec, as ``Benchmark.replace`` names
            # its result: ``irds:trec-covid[docs=irds:other-docs,...]``.
            base, _, spec = name[:-1].partition('[')
            facets = dict(pair.split('=', 1) for pair in spec.split(',') if pair)
            node = self[base]
            if not hasattr(node, 'replace'):
                raise KeyError(f'{base!r} does not support facet swaps: {name!r}')
            return node.replace(**facets)
        if ':' not in name:
            node, name = self._load_bare(name)
        else:
            node = self.provider_for(name).load(name)
        if is_subclass(node.type, _ALIAS_TYPE):
            targets = self.edges_of(name).get(_ALIAS_OF_KIND)
            if not targets:
                raise KeyError(
                    f'{name!r} is an alias with no known target')
            return self[targets[0]]
        return node

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def load(self, name):
        """``self[name]``, warning (``requirements.MissingDependencyWarning``) if the
        node needs packages that aren't installed."""
        node = self[name]
        requirements.warn_if_missing(name, self.requires(node))
        return node

    def requires(self, node):
        """Every extra package a node needs: its own, plus those of everything
        it is built from. ``node`` is a node or a name."""
        if isinstance(node, str):
            node = self[node]
        nodes, seen = [node], set()
        structural = structural_kinds()
        for edge in node.structural_edges():
            if edge.kind not in structural:       # an informational edge (a bundle's parts) is not a dependency
                continue
            target = edge.target if isinstance(edge.target, str) else edge.target.qualified_name
            for name in ([target, *self.closure(target)] if target else []):
                if name not in seen:
                    seen.add(name)
                    nodes.append(self[name])
        out = []
        for n in nodes:
            out.extend(getattr(n, 'requires', ()) or ())
        return tuple(dict.fromkeys(out))

    def generator_for(self, name):
        """A provider's ``Generator`` for a name, if it has one -- an optional
        capability (see ``registry.ManifestProvider``), not part of the
        minimal ``protocols.Provider`` contract."""
        try:
            provider = self.provider_for(name)
        except KeyError:
            return None
        return getattr(provider, 'generator_for', lambda n: None)(name)

    # -- traversal (union over providers; no imports needed) -----------------

    def edges_of(self, name, structural_only=False):
        """``{kind: [target, ...]}`` for a node, from every provider -- an
        informational edge about a node may live in another package.

        ``structural_only`` asks the node's own provider alone: structural edges are the owner's
        to declare (``add_edge`` refuses anyone else's), so no other provider can have one. That
        keeps build-graph questions -- everything ``load()`` asks, via ``requires`` -- from
        discovering the catalog of a provider that may be live (a crawl, a download)."""
        out = {}
        providers = self.providers.values()
        if structural_only:
            owner = self.providers.get(name.split(':', 1)[0]) if ':' in name else None
            providers = [owner] if owner is not None else providers
        for provider in providers:
            for kind, targets in self._catalog(provider)['fwd'].get(name, {}).items():
                bucket = out.setdefault(kind, [])
                known = set(bucket)
                bucket.extend(t for t in targets if t not in known)
        if structural_only:
            structural = structural_kinds()
            out = {k: v for k, v in out.items() if k in structural}
        return out

    def targets(self, name, kind):
        """The resolved nodes at the end of a node's edges of one kind."""
        return [self[t] for t in self.edges_of(name).get(kind, ())]

    def edge_provenance(self, src, kind, dst):
        """Which provider(s) contributed an edge."""
        return [p.prefix for p in self.providers.values()
                if dst in self._catalog(p)['fwd'].get(src, {}).get(kind, [])]

    def dependencies(self, name):
        """Direct structural targets of a node."""
        return sorted({t for targets in
                       self.edges_of(name, structural_only=True).values()
                       for t in targets})

    def referrers(self, name, kind=None):
        """Nodes with an edge of any kind (or of `kind`) pointing at `name`.
        Indexed, across every provider: "which datasets should cite this
        paper?" is ``referrers(article, 'citation')``."""
        pairs = set()
        for provider in self.providers.values():
            pairs |= self._catalog(provider)['rev'].get(name, set())
        return sorted({s for s, k in pairs if kind is None or k == kind})

    def incoming_edges(self, name):
        """``(subject, kind)`` pairs for every edge pointing at `name` --
        the reverse-direction counterpart to ``edges_of``, kind preserved
        (unlike ``referrers``, which drops it)."""
        pairs = set()
        for provider in self.providers.values():
            pairs |= self._catalog(provider)['rev'].get(name, set())
        return sorted(pairs)

    def dependents(self, name):
        """Nodes that structurally depend on this one: "what uses this corpus?"
        is an edge query, not a prefix grep."""
        structural = structural_kinds()
        pairs = set()
        for provider in self.providers.values():
            pairs |= self._catalog(provider)['rev'].get(name, set())
        return sorted({s for s, k in pairs if k in structural})

    def closure(self, name):
        """Every node reachable from this one via structural edges."""
        seen, stack = set(), [name]
        while stack:
            for target in self.dependencies(stack.pop()):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return sorted(seen)

    def names(self):
        out = set()
        for provider in self.providers.values():
            out |= set(self._catalog(provider)['types'])
        return out

    def find_cycles(self):
        """Cycles in the structural subgraph, which must be a DAG."""
        names = self.names()
        state, cycles = {}, []

        def visit(node, path):
            if state.get(node) == 'done':
                return
            if state.get(node) == 'active':
                cycles.append(path[path.index(node):] + [node])
                return
            state[node] = 'active'
            for target in self.dependencies(node):
                if target in names:
                    visit(target, path + [target])
            state[node] = 'done'

        for node in sorted(names):
            visit(node, [node])
        return cycles

    def check(self):
        """Validate the graph: acyclic, and no dangling targets. Both are hard
        problems: a cycle means the build order is
        undefined, a dangling edge means a node references something nothing
        provides."""
        problems = ['cycle: ' + ' -> '.join(c) for c in self.find_cycles()]
        known = self.names()
        for provider in self.providers.values():
            fwd = self._catalog(provider)['fwd']
            for src, kinds in fwd.items():
                for kind, dsts in kinds.items():
                    for dst in dsts:
                        if dst in known:
                            continue
                        # A target a generator can produce on demand is not
                        # dangling, just unbuilt -- but matching the template is
                        # not enough: a typo'd measure matches too. Resolve it.
                        if self.generator_for(dst) is not None:
                            try:
                                self[dst]
                                continue
                            except KeyError:
                                pass
                        problems.append(f'dangling: {src} --{kind}--> {dst}')
        problems += self._check_domains_and_ranges()
        return problems

    def _check_domains_and_ranges(self):
        """An edge whose property declares a ``domain`` or a ``range_type`` must start at a
        node of that class, and end at one (an end that is not in the graph is reported as
        dangling, not here)."""
        declared = {k: v for k, v in graph_properties().items()
                    if v['range'] == 'node' and (v.get('domain') or v.get('range_type'))}
        problems = []
        for provider in self.providers.values():
            for src, kinds in self._catalog(provider)['fwd'].items():
                for kind, dsts in kinds.items():
                    entry = declared.get(kind)
                    if entry is None:
                        continue
                    if entry['domain'] and not self._has_type(src, entry['domain']):
                        problems.append(f'domain: {src} --{kind}--> {dsts[0]}: {src} is not an {entry["domain"]}')
                    if entry['range_type']:
                        for dst in dsts:
                            types = self.types_of(dst)
                            if types and not any(is_subclass(t, entry['range_type']) for t in types):
                                problems.append(f'range: {src} --{kind}--> {dst}: {dst} is not an {entry["range_type"]}')
        return problems

    def _has_type(self, name, cls):
        return any(is_subclass(t, cls) for t in self.types_of(name))

    def export_triples(self, providers=None):
        """Every provider's nodes/edges as RDF-ready ``(subject, predicate,
        object)`` triples (``object`` a ``base.Literal`` for a property
        value, an unwrapped qualified name for an edge target) -- for
        materializing the graph into an external triplestore. ``providers``
        restricts this to a subset of prefixes; each provider's own
        ``discover_edges()`` decides how (from its imported modules, or, like
        ``hf``, via its own live crawl)."""
        for prefix, provider in self.providers.items():
            if providers is not None and prefix not in providers:
                continue
            yield from provider.discover_edges()

    # -- listing ------------------------------------------------------------

    def type_of(self, name):
        try:
            provider = self.provider_for(name)
        except KeyError:
            return None
        return self._catalog(provider)['types'].get(name)

    def types_of(self, name):
        """Every type (``rdf:type``) of ``name`` (``type_of`` is just the first)."""
        try:
            provider = self.provider_for(name)
        except KeyError:
            return []
        return list(self._catalog(provider)['all_types'].get(name, ()))

    def list(self, type=None):
        """Names from every provider's catalog (``discover_edges()``), without
        importing or downloading anything a node's own data would need.
        ``type`` is a qualified graph class (``irds:Qrels``) -- matched by
        subtype, so ``type='irds:Table'`` also finds
        ``irds:Qrels``/``irds:Docs``/... nodes (see
        ``vocabulary.is_subclass``). Every registered node is listed -- there's
        no "hidden"/internal-only tier; a node either exists in the graph,
        addressable, or it was never registered.

        Always the provider's full, current catalog: ``discover_edges()``
        always does full discovery (see ``protocols.Provider``), so for a
        provider whose catalog is genuinely live (``hf``, crawling the Hub)
        that's exactly what runs here -- once per ``Graph``, since the result
        is cached (see ``_catalog``).
        """
        out = []
        for provider in self.providers.values():
            all_types = self._catalog(provider)['all_types']
            for name, node_types in all_types.items():
                if type is not None and not any(is_subclass(t, type) for t in node_types):
                    continue
                out.append(name)
        return sorted(out)


# ── The default graph ────────────────────────────────────────────────────────

def discover(group=ENTRY_POINT_GROUP):
    """Providers declared by installed packages. The entry point's *name* must
    equal the provider's prefix: that is how a package claims a prefix."""
    from importlib.metadata import entry_points
    if sys.version_info >= (3, 10):
        eps = entry_points(group=group)
    else:  # pragma: no cover
        eps = entry_points().get(group, [])
    out = []
    for ep in eps:
        provider = ep.load()
        if provider.prefix != ep.name:
            raise ValueError(
                f'entry point {ep.name!r} ({ep.value}) provides prefix '
                f'{provider.prefix!r}; the entry point name must be the prefix')
        out.append(provider)
    return out


_DEFAULT = None


def default_graph():
    """The Graph over every installed entry point -- there is no default
    provider."""
    global _DEFAULT
    if _DEFAULT is None:
        # Assigned before discovery: loading an entry point imports its
        # package, which may itself call default_graph().add(...).
        _DEFAULT = Graph()
        for provider in discover():
            _DEFAULT.add(provider)
    return _DEFAULT
