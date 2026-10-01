"""``ManifestProvider``: a package's own registry, and one implementation of
``protocols.Provider``.

``Graph`` requires nothing more of a provider than ``protocols.Provider``:
a ``prefix``, ``load(name)``, and ``discover_edges()``. ``ManifestProvider``
is the batteries-included way to satisfy that for a package with a static
catalog worth freezing ahead of time -- registration, vocabulary declaration,
generators, and a frozen manifest all come with it. A provider
doesn't have to be one of these (see ``hf_provider.HfProvider``, which mixes
this in but overrides ``discover_edges`` for a live crawl instead of a
manifest), and a minimal third-party provider need not use it at all::

    # acme/__init__.py               [project.entry-points."ir_datasets.providers"]
    acme = ManifestProvider('acme')     # acme = "acme:acme"

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
edges about other providers' nodes), its generators, and its manifest -- and its
**vocabulary**: the node types and edge kinds it declares
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
from .base import Literal
from .vocabulary import (
    check_kind, check_type, declare_kind, declare_type, is_structural,
)

PREFIX = re.compile(r'^[A-Za-z0-9_\-]+$')

#: Loading a node by its old v1 id (``ir_datasets.v2.load('antique/test')``)
#: is not a per-provider concern any more -- see ``graph.py``'s own docstring
#: and ``Graph.__getitem__``. The ``legacy:`` provider (``legacy_provider.py``)
#: makes every v1 id -- across every bundled provider -- into a real,
#: addressable ``legacy:V1Dataset`` node with a ``legacy:replaced_by`` edge to
#: its v2 counterpart where one is known; ``Graph`` is what turns a lookup
#: that lands on one of those into its replacement (or fails, if there is
#: none).


class DuplicateNameError(LookupError):
    pass


def row_triples(name, row):
    """A manifest row (``freeze.row_for``'s shape, or a frozen row loaded from
    disk) as RDF-ready ``(subject, predicate, object)`` triples: one ``type``
    triple, then one triple per non-identity field -- repeated for each
    element of a list-valued field (RDF's natural multi-valued-property
    idiom), or a single JSON-string literal for a dict-valued one
    (``samples``, ``defs``: display blobs, not graph structure). Every field
    value is wrapped in ``base.Literal`` -- only a ``type`` object (another
    node's declared type, not a property value) is left as a plain qualified
    name. ``name``/``module`` are identity/implementation detail, never
    triples. Shared by ``ManifestProvider.discover_edges`` and any provider
    (e.g. ``hf``) that builds rows from live nodes instead of a frozen
    manifest."""
    if row.get('type'):
        yield name, 'type', row['type']
    for field, value in row.items():
        if field in ('name', 'type', 'module'):
            continue
        if isinstance(value, list):
            for item in value:
                yield name, field, Literal(item)
        elif isinstance(value, dict):
            yield name, f'{field}_json', Literal(json.dumps(value, sort_keys=True))
        else:
            yield name, field, Literal(value)


def _caller_module(depth=2):
    """Module of the code that called the public API (one frame up from it)."""
    try:
        return sys._getframe(depth).f_globals.get('__name__')
    except ValueError:
        return None


class ManifestProvider:
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
        # Vocabulary this provider owns.
        self.types = {}            # qualified type -> {'desc', 'parent'}
        self.edge_kinds = {}       # qualified kind -> {'structural', 'desc'}
        self.defaultable_fields = set()
        # Edges this provider contributes: src -> kind -> [dst]; dst -> {(src, kind)}.
        self._fwd = {}
        self._rev = {}
        self._manifest = None
        self._manifest_index = None
        self._bootstrapped = False

    def __repr__(self):
        return f'{type(self).__name__}({self.prefix!r})'

    def qualify(self, name):
        """Apply this provider's prefix to a bare name; qualified names pass."""
        return name if ':' in name else f'{self.prefix}:{name}'

    def owns(self, name):
        """Whether a qualified name falls in this provider's namespace."""
        return name.startswith(f'{self.prefix}:')

    # -- vocabulary ---------------------------------------------------------

    def node_type(self, name, *, desc=None, parent=None):
        """Declare a node type this provider owns; returns the qualified type
        for a Node subclass to set as ``type``::

            class DocTable(Table):
                type = irds.node_type('DocTable', parent=TABLE)

        ``parent`` is another qualified type -- this provider's own, or
        another provider's entirely (a third-party ``ext:MyTable`` may set
        ``parent='irds:Table'``) -- making this one a subtype of it for
        ``is_subtype``/``Graph.list(type=...)`` purposes. The type itself is
        still owned (and only ever registerable) by this provider; ``parent``
        is a graph relationship, not a transfer of ownership.
        """
        qualified = self.qualify(name)
        if not self.owns(qualified):
            raise ValueError(f'{self} cannot declare the type {name!r}')
        self.types[qualified] = {'desc': desc, 'parent': parent}
        return declare_type(qualified, owner=self.prefix, desc=desc, parent=parent)

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
                                  'types': {}, 'edge_kinds': {}, 'defaultable': []}
            self._manifest_index = None
            # A frozen provider's vocabulary is known without importing it.
            for qualified, entry in self._manifest.get('types', {}).items():
                if self.owns(qualified):
                    declare_type(qualified, owner=self.prefix, desc=entry.get('desc'),
                                parent=entry.get('parent'))
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

    def type_of(self, name):
        node = self.nodes.get(name)
        if node is not None:
            return node.type
        return self.frozen(name).get('type')

    # -- protocols.Provider ---------------------------------------------------

    def discover_edges(self):
        """This provider's whole current catalog, as RDF-ready
        ``(subject, predicate, object)`` triples (``object`` wrapped in
        ``base.Literal`` when it's a property value rather than another
        node's name -- see ``row_triples``) -- live-registered nodes (not yet
        frozen) merged with whatever the
        frozen manifest already has, plus every edge either contributes.
        Never triggers a generator expansion (a parametric family is a rule,
        not a catalog entry -- see ``freeze.build_manifest``'s own
        ``generated`` skip). Bootstraps first (imports every module) if
        nothing has been frozen yet, so an editable checkout with no
        ``manifest.json`` still reports its full catalog; a subclass with a
        genuinely live catalog (``hf_provider.HfProvider``) overrides this
        entirely instead of relying on a manifest at all.
        """
        from .freeze import RETIRED_FIELDS, row_for
        # Always import the package's modules, not only when nothing is frozen:
        # facts derived from a node's own declaration (size, validation,
        # sources -- see ``Node.discovery_literals``) are reported from the
        # live node, never written to the manifest.
        self.import_all()
        seen = set()
        for name, node in self.nodes.items():
            if name in self.generated:
                continue
            # A node needs no particular base class (see protocols.Node), so
            # discovery_literals is optional.
            literals = getattr(node, 'discovery_literals', lambda: {})()
            row = {**row_for(node), **{k: v for k, v in literals.items() if v}}
            # The live declaration wins, but what ``freeze --verify`` attested
            # (count, content hash, samples, score counts, ...) only exists in
            # the manifest: carry it over, minus fields that are now derived
            # from the declaration instead (RETIRED_FIELDS).
            row = {**{k: v for k, v in self.manifest()['nodes'].get(name, {}).items()
                      if k not in RETIRED_FIELDS and k not in row}, **row}
            if getattr(node, 'samples_permitted', True) is False:
                row.pop('samples', None)  # see freeze._attested_row
            yield from row_triples(name, row)
            seen.add(name)
        for name, row in self.manifest()['nodes'].items():
            if name not in seen:
                yield from row_triples(name, row)
        fwd, _ = self.indexes()
        for src, kinds in fwd.items():
            for kind, dsts in kinds.items():
                for dst in dsts:
                    yield src, kind, dst
        types = {**self.manifest()['types'], **self.types}
        for qualified, entry in types.items():
            if self.owns(qualified) and entry.get('parent'):
                yield qualified, 'subClassOf', entry['parent']
        # Enumerable generators (every param has a closed values= set, e.g.
        # CLIRMatrix's languages/splits) report a type row per name they
        # could produce, plus whatever structural edges the generator
        # declared (see Generator.edges) -- e.g. a Benchmark generator
        # naming its docs/queries/qrels facets, or a Table generator naming
        # the Resource it's parsed from. All by name-template substitution,
        # cheap (no resolution, no download, no network) -- so a listing
        # sees the whole family's real shape without visiting each member
        # first. A non-enumerable one (hf's single open-ended pattern= param)
        # has no finite cross product to report here; that provider
        # discovers its own names some other way (see
        # HfProvider.discover_edges).
        #
        # A generator that also declared row_metadata= (e.g. CLIRMatrix's
        # per-file Resource generators, backed by one cached remote index --
        # see datasets/clirmatrix.py) reports real per-name properties
        # (sources, hashes) here too, via enumerate_rows() instead of the
        # bare type row -- still no per-name network cost as long as
        # row_metadata amortizes its own work (see Generator.row_metadata's
        # docstring); a generator without one keeps reporting bare type rows.
        for generator in self.generators:
            if generator.enumerable:
                if generator.row_metadata is not None:
                    for name, row in generator.enumerate_rows():
                        yield from row_triples(name, row)
                else:
                    for name in generator.enumerate():
                        yield name, 'type', generator.node_type
                yield from generator.enumerate_edges()

    # -- lookup -------------------------------------------------------------

    def load(self, name):
        """Resolve one of this provider's (already-qualified) names to a
        node -- the ``protocols.Provider`` entry point; ``__getitem__``
        additionally accepts a bare, unqualified name."""
        return self[name]

    def __getitem__(self, name):
        """Resolve one of this provider's nodes by (qualified or bare) name."""
        name = self.qualify(name)
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
