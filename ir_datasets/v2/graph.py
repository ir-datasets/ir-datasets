"""``Graph``: the knowledge graph assembled from every installed provider.

Nothing registers *into* a Graph, and a Graph asks nothing of a provider
beyond ``protocols.Provider``: ``load(name)`` to resolve one node, and
``discover_edges()`` to get its whole catalog as RDF-ready
``(subject, predicate, object)`` triples -- ``object`` wrapped in
``base.Literal`` when it's a property value rather than another node's name
(see ``registry.row_triples``). Everything
below -- traversal, listing, export, validation, even legacy-alias
resolution (an alias is just an ``irds:alias`` edge, discovered like any
other) -- is built from those two calls. A provider's catalog is cached the
first time a ``Graph`` reads it, since ``discover_edges()`` always does full
discovery and may be a live crawl (see ``hf``) -- so a query doesn't repeat
that work, but only for as long as this particular ``Graph`` lives.

``default_graph()`` is the one over every installed entry point in the
``ir_datasets.providers`` group. This package declares no providers of its own
beyond ``irds`` (added explicitly in ``ir_datasets/v2/__init__.py``). A package
may also add its provider directly when it is imported (``default_graph().add``),
which is what makes an editable checkout work before its entry points exist.

There is no default provider. Every name is ``prefix:name``; a bare name
resolves only if some provider declared it as a (legacy) alias.
"""
import sys

from .base import Literal
from .registry import ALIAS_KIND
from .vocabulary import is_subtype, structural_kinds

ENTRY_POINT_GROUP = 'ir_datasets.providers'


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
        """``{'types', 'aliases', 'fwd', 'rev'}`` for one provider, built once
        from ``discover_edges()`` and cached -- everything else in this class
        reads from here rather than calling the provider again."""
        cached = self._catalogs.get(provider.prefix)
        if cached is None:
            types, aliases, fwd, rev = {}, {}, {}, {}
            for subject, kind, obj in provider.discover_edges():
                if isinstance(obj, Literal):
                    continue
                if kind == 'type':
                    types[subject] = obj
                elif kind == ALIAS_KIND:
                    aliases[subject] = obj
                elif kind == 'subClassOf':
                    # Type-hierarchy metadata (vocabulary.py's own
                    # declare_type/ancestors already track this for
                    # is_subtype) -- not a node-to-node edge, so it must not
                    # show up in edges_of/check's dangling-target search.
                    continue
                else:
                    bucket = fwd.setdefault(subject, {}).setdefault(kind, [])
                    if obj not in bucket:
                        bucket.append(obj)
                    rev.setdefault(obj, set()).add((subject, kind))
            cached = {'types': types, 'aliases': aliases, 'fwd': fwd, 'rev': rev}
            self._catalogs[provider.prefix] = cached
        return cached

    # -- names --------------------------------------------------------------

    def resolve_name(self, name):
        """A legacy alias -> its qualified target; anything else unchanged."""
        for provider in self.providers.values():
            aliases = self._catalog(provider)['aliases']
            if name in aliases:
                return aliases[name]
        return name

    def provider_for(self, name):
        """The provider whose namespace a qualified name falls in."""
        if ':' not in name:
            raise KeyError(
                f'{name!r} is not a qualified name and not a known alias; every '
                f'node lives in a provider (<prefix>:{name}; installed prefixes: '
                f'{", ".join(sorted(self.providers)) or "none"})')
        prefix = name.split(':', 1)[0]
        try:
            return self.providers[prefix]
        except KeyError:
            raise KeyError(
                f'no provider for prefix {prefix!r} (needed by {name!r}); is '
                f'the package that provides it installed?') from None

    def __getitem__(self, name):
        name = self.resolve_name(name)
        return self.provider_for(name).load(name)

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def load(self, name):
        return self[name]

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
        informational edge about a node may live in another package."""
        name = self.resolve_name(name)
        out = {}
        for provider in self.providers.values():
            for kind, targets in self._catalog(provider)['fwd'].get(name, {}).items():
                bucket = out.setdefault(kind, [])
                bucket.extend(t for t in targets if t not in bucket)
        if structural_only:
            structural = structural_kinds()
            out = {k: v for k, v in out.items() if k in structural}
        return out

    def targets(self, name, kind):
        """The resolved nodes at the end of a node's edges of one kind."""
        return [self[t] for t in self.edges_of(name).get(kind, ())]

    def edge_provenance(self, src, kind, dst):
        """Which provider(s) contributed an edge."""
        src, dst = self.resolve_name(src), self.resolve_name(dst)
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
        name = self.resolve_name(name)
        pairs = set()
        for provider in self.providers.values():
            pairs |= self._catalog(provider)['rev'].get(name, set())
        return sorted({s for s, k in pairs if kind is None or k == kind})

    def incoming_edges(self, name):
        """``(subject, kind)`` pairs for every edge pointing at `name` --
        the reverse-direction counterpart to ``edges_of``, kind preserved
        (unlike ``referrers``, which drops it)."""
        name = self.resolve_name(name)
        pairs = set()
        for provider in self.providers.values():
            pairs |= self._catalog(provider)['rev'].get(name, set())
        return sorted(pairs)

    def dependents(self, name):
        """Nodes that structurally depend on this one: "what uses this corpus?"
        is an edge query, not a prefix grep."""
        name = self.resolve_name(name)
        structural = structural_kinds()
        pairs = set()
        for provider in self.providers.values():
            pairs |= self._catalog(provider)['rev'].get(name, set())
        return sorted({s for s, k in pairs if k in structural})

    def closure(self, name):
        """Every node reachable from this one via structural edges."""
        name = self.resolve_name(name)
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
        """Validate the graph: acyclic, and no dangling targets. Hard problems,
        unlike a missing attestation: a cycle means the build order is
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
        return problems

    # -- frozen data --------------------------------------------------------

    def frozen(self, name):
        """A node's frozen row, from its own provider -- an optional
        capability (a manifest-backed provider has one; a minimal
        ``protocols.Provider`` need not)."""
        name = self.resolve_name(name)
        try:
            provider = self.provider_for(name)
        except KeyError:
            return {}
        frozen_fn = getattr(provider, 'frozen', None)
        return frozen_fn(name) if frozen_fn else {}

    def export_triples(self, providers=None):
        """Every provider's nodes/edges as RDF-ready ``(subject, predicate,
        object)`` triples (``object`` a ``base.Literal`` for a property
        value, an unwrapped qualified name for an edge target) -- for
        materializing the graph into an external triplestore. ``providers``
        restricts this to a subset of prefixes; each provider's own
        ``discover_edges()`` decides how (from a manifest, or, like ``hf``,
        via its own live crawl)."""
        for prefix, provider in self.providers.items():
            if providers is not None and prefix not in providers:
                continue
            yield from provider.discover_edges()

    # -- listing ------------------------------------------------------------

    def type_of(self, name):
        name = self.resolve_name(name)
        try:
            provider = self.provider_for(name)
        except KeyError:
            return None
        return self._catalog(provider)['types'].get(name)

    def list(self, type=None):
        """Names from every provider's catalog (``discover_edges()``), without
        importing or downloading anything a node's own data would need.
        ``type`` is a qualified node type (``irds:QrelTable``) -- matched by
        subtype, so ``type='irds:Table'`` also finds
        ``irds:QrelTable``/``irds:DocTable``/... nodes (see
        ``vocabulary.is_subtype``). Every registered node is listed -- there's
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
            types = self._catalog(provider)['types']
            for name, node_type in types.items():
                if type is not None and not is_subtype(node_type, type):
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
