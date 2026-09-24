"""``Provider``: a package's own registry.

A provider is the unit of decentralization. Each package that contributes
nodes has one, declares it via an entry point whose *name* is the prefix it
owns, and registers its nodes into it::

    # acme/__init__.py               [project.entry-points."ir_datasets.providers"]
    acme = Provider('acme')             # acme = "acme:acme"

    # acme/things.py
    from acme import acme
    ...
    acme.register(train, test)          # dependencies come along (see register)
    acme.add_edges([docs, train, test], 'citation', ARTICLES)

Registration is explicit. Constructing a node creates an object and nothing
else, so ad-hoc nodes (a local TSV, a test fixture) never touch any registry.
The prefix is intrinsic to the provider object -- there is no ambient "current
provider" state anywhere.

A provider owns: its nodes, the edges *it* contributes (including informational
edges about other providers' nodes), its generators, its legacy aliases, and its
manifest -- and its **vocabulary**: the node types and edge kinds it declares
(``irds:docs``, ``irds:derived_from``) and the fields it allows ``defaults()`` to set.
Everything in the graph has a home, and the home is in the name. The union over
all installed providers is a ``Graph``.
"""
import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path

from . import context
from .vocabulary import (
    check_kind, check_type, declare_kind, declare_type, is_structural,
)

PREFIX = re.compile(r'^[A-Za-z0-9_\-]+$')


class DuplicateNameError(LookupError):
    pass


def _caller_module(depth=2):
    """Module of the code that called the public API (one frame up from it)."""
    try:
        return sys._getframe(depth).f_globals.get('__name__')
    except ValueError:
        return None


class Provider:
    def __init__(self, prefix, *, package=None, manifest_path=None):
        if not PREFIX.match(prefix or ''):
            raise ValueError(f'invalid provider prefix {prefix!r}')
        self.prefix = prefix
        #: Dotted path of the package whose modules register into this provider.
        #: Used to import everything at freeze time, and to bootstrap before a
        #: manifest exists.
        self.package = package
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.nodes = {}            # qualified name -> node
        self.generated = set()     # names of nodes a generator produced
        self.generators = []
        self.aliases = {}          # legacy id -> qualified name
        # Vocabulary this provider owns.
        self.types = {}            # qualified type -> {'desc'}
        self.edge_kinds = {}       # qualified kind -> {'structural', 'desc'}
        self.defaultable_fields = set()
        # Edges this provider contributes: src -> kind -> [dst]; dst -> {(src, kind)}.
        self._fwd = {}
        self._rev = {}
        self._manifest = None
        self._manifest_index = None
        self._bootstrapped = False
        self._known_fn = None

    def __repr__(self):
        return f'Provider({self.prefix!r})'

    def qualify(self, name):
        """Apply this provider's prefix to a bare name; qualified names pass."""
        return name if ':' in name else f'{self.prefix}:{name}'

    def owns(self, name):
        """Whether a qualified name falls in this provider's namespace."""
        return name.startswith(f'{self.prefix}:')

    # -- vocabulary ---------------------------------------------------------

    def node_type(self, name, *, desc=None):
        """Declare a node type this provider owns; returns the qualified type
        for a Node subclass to set as ``type``::

            class Docs(Table):
                type = irds.node_type('Docs')
        """
        qualified = self.qualify(name)
        if not self.owns(qualified):
            raise ValueError(f'{self} cannot declare the type {name!r}')
        self.types[qualified] = {'desc': desc}
        return declare_type(qualified, owner=self.prefix, desc=desc)

    def edge_kind(self, name, *, structural, desc=None):
        """Declare an edge kind this provider owns. Structural kinds form the
        build DAG; informational kinds are anything else."""
        qualified = self.qualify(name)
        if not self.owns(qualified):
            raise ValueError(f'{self} cannot declare the edge kind {name!r}')
        self.edge_kinds[qualified] = {'structural': structural, 'desc': desc}
        return declare_kind(qualified, structural=structural, owner=self.prefix,
                            desc=desc)

    def defaultable(self, *fields):
        """Declare node fields this provider's ``defaults()`` may set.
        Descriptive fields only -- never identity or data."""
        self.defaultable_fields.update(fields)
        return fields

    def defaults(self, **kwargs):
        """Fill in unset node fields for everything constructed in the block."""
        unknown = sorted(set(kwargs) - self.defaultable_fields)
        if unknown:
            raise ValueError(
                f'{self} cannot default {", ".join(unknown)}; declared: '
                f'{", ".join(sorted(self.defaultable_fields)) or "(none)"}')
        return context.push(kwargs)

    # -- registration -------------------------------------------------------

    def register(self, *nodes, module=None):
        """Add nodes to this provider.

        Stamps ``qualified_name`` (prefix applied), ``provider`` and
        ``defined_in`` (the calling module, or ``module=``). Ingests each node's
        ``structural_edges()``; a dependency given as an *object* that is not yet
        registered is registered too, transitively -- a registered node with an
        unregistered dependency would be an invalid graph, so the module need
        only register its roots. Dependencies given as *names* are assumed to be
        provided elsewhere and are checked at freeze time. A bare name or edge
        kind means this provider's own namespace -- the one shorthand, and only
        inside a provider's own modules, where the provider is in plain sight.
        """
        module = module or _caller_module()
        for node in nodes:
            self._register_one(node, module)
        return nodes

    def _register_one(self, node, module):
        missing = [m for m in ('structural_edges', 'attest', 'verify')
                  if not callable(getattr(node, m, None))]
        if missing:
            raise TypeError(
                f'{node!r} does not look like a Node: missing {", ".join(missing)}(). '
                f'A node needs no particular base class (see protocols.Node) -- '
                f'just this shape.')
        if node.provider is self and node.qualified_name in self.nodes:
            return
        if node.provider is not None and node.provider is not self:
            raise ValueError(
                f'{node.name!r} is already registered with {node.provider}')
        check_type(node.type)
        name = self.qualify(node.name)
        existing = self.nodes.get(name)
        if existing is not None and existing is not node:
            # Re-importing a module rebuilds equivalent nodes; only complain
            # when two *different* definitions claim one name.
            if existing.defined_in != module:
                raise DuplicateNameError(
                    f'{name!r} is defined by both {existing.defined_in} and {module}')
        node.qualified_name = name
        node.provider = self
        node.defined_in = module
        self.nodes[name] = node
        for edge in node.structural_edges():
            kind = check_kind(self.qualify(edge.kind))
            target = edge.target
            if isinstance(target, str):
                target = self.qualify(target)
            else:
                if target.provider is None:
                    self._register_one(target, module)
                target = target.qualified_name
            self._put_edge(name, kind, target)

    def register_generator(self, generator, module=None):
        check_type(generator.node_type)
        generator.provider = self
        generator.defined_in = module or _caller_module()
        self.generators.append(generator)
        return generator

    def alias(self, legacy_name, name=None):
        """Map legacy ids onto this provider's node names -- a mapping, or a
        single pair. Aliases are permanent: old ids appear in published papers."""
        items = legacy_name.items() if name is None else [(legacy_name, name)]
        for legacy, target in items:
            self.aliases[legacy] = self.qualify(target)

    # -- edges --------------------------------------------------------------

    def _name(self, node_or_name):
        if isinstance(node_or_name, str):
            return self.qualify(node_or_name)
        if node_or_name.qualified_name is None:
            raise ValueError(f'{node_or_name!r} is not registered')
        return node_or_name.qualified_name

    def _put_edge(self, src, kind, dst):
        bucket = self._fwd.setdefault(src, {}).setdefault(kind, [])
        if dst not in bucket:
            bucket.append(dst)
        self._rev.setdefault(dst, set()).add((src, kind))

    def add_edge(self, src, kind, dst):
        """Contribute an edge. Authority follows the structural/informational
        split: structural edges define what a node *is*, so only its own
        provider may declare them; informational edges (``citation``,
        ``see_also``, ...) describe a node, so any provider may add them to any
        node -- they ship in *this* provider's manifest."""
        kind = check_kind(self.qualify(kind))
        src, dst = self._name(src), self._name(dst)
        if is_structural(kind) and src not in self.nodes:
            raise PermissionError(
                f'{kind!r} is a structural edge; only the provider that owns '
                f'{src!r} may declare it, not {self}')
        self._put_edge(src, kind, dst)

    def add_edges(self, srcs, kind, dsts):
        """``add_edge`` for every (src, dst) pair."""
        for src in srcs:
            for dst in dsts:
                self.add_edge(src, kind, dst)

    def edge_rows(self):
        """This provider's in-memory edges as ``[src, kind, dst]`` rows."""
        return sorted([s, k, d] for s, kinds in self._fwd.items()
                      for k, dsts in kinds.items() for d in dsts)

    def indexes(self):
        """(forward, reverse) over this provider's edges: manifest + memory."""
        fwd, rev = self._manifest_indexes()
        fwd = {s: {k: list(v) for k, v in kinds.items()} for s, kinds in fwd.items()}
        rev = {d: set(v) for d, v in rev.items()}
        for src, kinds in self._fwd.items():
            for kind, dsts in kinds.items():
                bucket = fwd.setdefault(src, {}).setdefault(kind, [])
                bucket.extend(d for d in dsts if d not in bucket)
        for dst, pairs in self._rev.items():
            rev.setdefault(dst, set()).update(pairs)
        return fwd, rev

    # -- manifest -----------------------------------------------------------

    def manifest(self):
        if self._manifest is None:
            if self.manifest_path and self.manifest_path.exists():
                with open(self.manifest_path) as fin:
                    self._manifest = json.load(fin)
            else:
                self._manifest = {'nodes': {}, 'edges': [], 'generators': [],
                                  'aliases': {}, 'types': {},
                                  'edge_kinds': {}, 'defaultable': []}
            self._manifest_index = None
            for legacy, target in self._manifest.get('aliases', {}).items():
                self.aliases.setdefault(legacy, target)
            # A frozen provider's vocabulary is known without importing it.
            for qualified, entry in self._manifest.get('types', {}).items():
                if self.owns(qualified):
                    declare_type(qualified, owner=self.prefix, desc=entry.get('desc'))
            for qualified, entry in self._manifest.get('edge_kinds', {}).items():
                if self.owns(qualified):
                    declare_kind(qualified, structural=entry['structural'],
                                 owner=self.prefix, desc=entry.get('desc'))
        return self._manifest

    def _manifest_indexes(self):
        if self._manifest_index is None:
            fwd, rev = {}, {}
            for src, kind, dst in self.manifest().get('edges', []):
                fwd.setdefault(src, {}).setdefault(kind, []).append(dst)
                rev.setdefault(dst, set()).add((src, kind))
            self._manifest_index = (fwd, rev)
        return self._manifest_index

    def frozen(self, name):
        """The frozen manifest row for one of this provider's nodes."""
        return self.manifest()['nodes'].get(name, {})

    def names(self):
        return set(self.nodes) | set(self.manifest()['nodes'])

    def register_known(self, fn, module=None):
        """Register a callable returning this provider's known-but-unresolved
        names -- for a dynamic provider with no manifest, a live/expensive way
        to advertise what's out there without resolving any of it. Optional; a
        provider with nothing to add here just has no known_names(). Not
        merged into names()/__contains__ -- see Graph.list(discover=...)."""
        self._known_fn = fn
        return fn

    def known_names(self):
        """Qualified names from register_known's callable, or empty if unset."""
        if self._known_fn is None:
            return set()
        return {self.qualify(n) for n in self._known_fn()}

    def type_of(self, name):
        node = self.nodes.get(name)
        if node is not None:
            return node.type
        return self.frozen(name).get('type')

    # -- lookup -------------------------------------------------------------

    def __getitem__(self, name):
        """Resolve one of this provider's nodes by (qualified or bare) name."""
        name = self.aliases.get(name, self.qualify(name))
        if name in self.nodes:
            return self.nodes[name]
        # Import the ONE module that defines it, per the manifest.
        row = self.frozen(name)
        if row:
            importlib.import_module(row['module'])
            if name in self.nodes:
                return self.nodes[name]
            raise KeyError(
                f'{name!r} is frozen as defined in {row["module"]}, but importing '
                f'it did not register the node; re-run freeze')
        # Parametric / dynamic: build it.
        for generator in self.generators:
            params = generator.match(name)
            if params is not None:
                try:
                    node = generator.resolve_node(params)
                except (ValueError, NameError, LookupError) as e:
                    # The template matched but the resolver rejected the
                    # parameters (an unknown measure, say): that is "no such
                    # node", not an internal error. ImportError etc. propagate.
                    raise KeyError(f'no such node: {name!r} ({e})') from e
                self.register(node, module=generator.defined_in)
                self.generated.add(node.qualified_name)
                return node
        # No manifest yet: import everything once and retry.
        if self._bootstrap():
            return self[name]
        raise KeyError(f'no such node: {name!r}')

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def generator_for(self, name):
        for generator in self.generators:
            if generator.match(name) is not None:
                return generator
        return None

    def import_all(self):
        """Import every module in this provider's package (freeze, bootstrap)."""
        if not self.package:
            return
        module = importlib.import_module(self.package)
        for info in pkgutil.iter_modules(module.__path__):
            importlib.import_module(f'{self.package}.{info.name}')

    def _bootstrap(self):
        if self._bootstrapped or not self.package or self.manifest()['nodes']:
            return False
        self._bootstrapped = True
        self.import_all()
        return True
