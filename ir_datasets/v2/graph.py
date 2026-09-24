"""``Graph``: the knowledge graph assembled from every installed provider.

Nothing registers *into* a Graph. It reads providers: routes lookups by prefix,
unions their edge indexes so a reverse lookup sees edges from every package (an
extension's ``see_also`` about another package's node; every dataset citing a
paper), enumerates and searches, and validates the whole.

``default_graph()`` is the one over every installed entry point in the
``ir_datasets.providers`` group. This package declares no providers of its own
beyond ``irds`` (added explicitly in ``ir_datasets/v2/__init__.py``). A package
may also add its provider directly when it is imported (``default_graph().add``),
which is what makes an editable checkout work before its entry points exist.

There is no default provider. Every name is ``prefix:name``; a bare name
resolves only if some provider declared it as a (legacy) alias.
"""
import sys

from .vocabulary import structural_kinds

ENTRY_POINT_GROUP = 'ir_datasets.providers'


class Graph:
    def __init__(self, providers=()):
        self.providers = {}     # prefix -> Provider
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

    # -- names --------------------------------------------------------------

    def resolve_name(self, name):
        """A legacy alias -> its qualified target; anything else unchanged."""
        for provider in self.providers.values():
            provider.manifest()  # loads frozen aliases
            if name in provider.aliases:
                return provider.aliases[name]
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
        return self.provider_for(name)[name]

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def load(self, name):
        return self[name]

    def generator_for(self, name):
        try:
            return self.provider_for(name).generator_for(name)
        except KeyError:
            return None

    # -- traversal (union over providers; no imports needed) -----------------

    def edges_of(self, name, structural_only=False):
        """``{kind: [target, ...]}`` for a node, from every provider -- an
        informational edge about a node may live in another package."""
        name = self.resolve_name(name)
        out = {}
        for provider in self.providers.values():
            for kind, targets in provider.indexes()[0].get(name, {}).items():
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
                if dst in p.indexes()[0].get(src, {}).get(kind, [])]

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
            pairs |= provider.indexes()[1].get(name, set())
        return sorted({s for s, k in pairs if kind is None or k == kind})

    def incoming_edges(self, name):
        """``(subject, kind)`` pairs for every edge pointing at `name` --
        the reverse-direction counterpart to ``edges_of``, kind preserved
        (unlike ``referrers``, which drops it)."""
        name = self.resolve_name(name)
        pairs = set()
        for provider in self.providers.values():
            pairs |= provider.indexes()[1].get(name, set())
        return sorted(pairs)

    def dependents(self, name):
        """Nodes that structurally depend on this one: "what uses this corpus?"
        is an edge query, not a prefix grep."""
        name = self.resolve_name(name)
        structural = structural_kinds()
        pairs = set()
        for provider in self.providers.values():
            pairs |= provider.indexes()[1].get(name, set())
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
            out |= provider.names()
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
            fwd, _ = provider.indexes()
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
        """A node's frozen row, from its own provider."""
        name = self.resolve_name(name)
        try:
            return self.provider_for(name).frozen(name)
        except KeyError:
            return {}

    # -- listing ------------------------------------------------------------

    def type_of(self, name):
        name = self.resolve_name(name)
        try:
            return self.provider_for(name).type_of(name)
        except KeyError:
            return None

    def list(self, type=None, discover=False):
        """Names from every provider, without importing their modules. ``type``
        is a qualified node type (``irds:docs``). Every registered node is
        listed -- there's no "hidden"/internal-only tier; a node either exists
        in the graph, addressable, or it was never registered.

        ``discover=True`` additionally asks each provider for its
        ``known_names()`` -- for a dynamic provider (e.g. ``hf``) with nothing
        to bootstrap or freeze locally, a live/expensive lookup of what's out
        there. Off by default since it's not a free operation like the rest
        of this method. A provider's ``known_names()`` may or may not resolve
        what it finds as a side effect (``hf``'s does, so its entries come
        back with a real type); one that only names things without resolving
        them would have those silently excluded by a ``type=`` filter, since
        ``provider.type_of`` falls through to ``None`` for anything not
        actually registered.
        """
        out = []
        for provider in self.providers.values():
            if not provider.manifest()['nodes']:
                provider._bootstrap()
            names = set(provider.names())
            if discover:
                names |= provider.known_names()
            for name in names:
                if type is not None and provider.type_of(name) != type:
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
