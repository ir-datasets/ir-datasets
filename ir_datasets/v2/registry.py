"""``ManifestProvider``: a package's own registry, and one implementation of
``protocols.Provider``.

``Graph`` requires nothing more of a provider than ``protocols.Provider``:
a ``prefix``, ``load(name)``, and ``discover_edges()``. ``ManifestProvider``
is the batteries-included way to satisfy that for a package with a static
catalog -- registration, vocabulary declaration and generators all come with
it. A provider
doesn't have to be one of these (see ``hf_provider.HfProvider``, which mixes
this in but overrides ``discover_edges`` for a live crawl), and a minimal third-party provider need not use it at all::

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
edges about other providers' nodes), its and generators -- and its
**vocabulary**: the graph classes and edge kinds it declares
(``irds:docs``, ``irds:derivedFrom``) and the fields it allows ``defaults()`` to set.
Everything in the graph has a home, and the home is in the name. The union over
all installed providers is a ``Graph``.
"""
import importlib
import json
import pkgutil
import re
import sys

from .base import Literal
from . import protocols
from .vocabulary import (
    RDF_TYPE, RDFS_SUBCLASS_OF, canonical_json, check_kind, check_graph_class, declare_graph_property, declare_graph_class,
    is_structural, property_for, type_for_protocol, types_of, warn_undeclared,
)

PREFIX = re.compile(r'^[A-Za-z0-9_\-]+$')

#: Loading a node by its old v1 id (``ir_datasets.v2.load('antique/test')``)
#: is not a per-provider concern any more -- see ``graph.py``'s own docstring
#: and ``Graph.__getitem__``. The ``legacy:`` provider (``legacy_provider.py``)
#: makes every v1 id -- across every bundled provider -- into a real,
#: addressable ``legacy:V1Dataset`` node with an ``irds:aliasOf`` edge to
#: its v2 counterpart where one is known; ``Graph`` is what turns a lookup
#: that lands on one of those into its replacement (or fails, if there is
#: none).


class DuplicateNameError(LookupError):
    pass


def row_triples(name, row):
    """A node's row (``row_for``'s shape) as RDF-ready ``(subject, predicate, object)`` triples: one
    ``rdf:type`` triple per type, then one triple per non-identity field, under
    that field's *declared* property (``provider.declare_graph_property(field, range=...)``,
    qualified by the owning provider: ``irds:description``). A field no provider
    has declared is left out, with an ``UndeclaredPropertyWarning``. A
    list value gets one triple per element (RDF has no single-valued
    properties); a ``json`` property's value is a JSON literal (a dict is
    serialized, a string is taken as already JSON). Every value is a ``base.Literal`` -- only a type object
    (another node's declared type, not a property value) is a plain qualified
    name. ``name``/``module`` are identity/implementation detail, never
    triples. Shared by ``ManifestProvider.discover_edges`` and any provider
    (e.g. ``hf``) that builds rows from live nodes."""
    from .vocabulary import graph_properties
    if row.get('type'):
        # A node with several types (``Benchmark.types``) has a list.
        for node_type in ([row['type']] if isinstance(row['type'], str) else row['type']):
            yield name, RDF_TYPE, node_type
    prefix = name.split(':', 1)[0]
    declared = graph_properties()
    for field, value in row.items():
        if field in ('name', 'type', 'module'):
            continue
        predicate = property_for(field, prefix)
        if predicate is None:
            warn_undeclared(field, name)
            continue
        is_json = declared[predicate]['range'] == 'json'
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if is_json:
                text = item if isinstance(item, str) else canonical_json(item)
                yield name, predicate, Literal(text, 'json')
            else:
                yield name, predicate, Literal(item)


def _caller_module(depth=2):
    """Module of the code that called the public API (one frame up from it)."""
    try:
        return sys._getframe(depth).f_globals.get('__name__')
    except ValueError:
        return None


class ManifestProvider:
    def __init__(self, prefix, *, package=None, distribution=None):
        if not PREFIX.match(prefix or ''):
            raise ValueError(f'invalid provider prefix {prefix!r}')
        self.prefix = prefix
        #: Dotted path of the package whose modules register into this provider.
        #: Imported in full the first time a name isn't found, or the whole
        #: catalog is wanted (``names``, ``indexes``, ``discover_edges``).
        self.package = package
        #: The pip distribution these nodes come from, reported as each node's
        #: ``package`` when it can't be told from the defining module alone.
        self.distribution = distribution
        self.nodes = {}            # qualified name -> node
        self.generated = set()     # names of nodes a generator produced
        self.generators = []
        # Vocabulary this provider owns.
        self.graph_classes = {}            # qualified type -> {'desc', 'parent'}
        self.graph_properties = {}       # qualified graph property -> {'range', 'structural', 'desc'}
        # Edges this provider contributes: src -> kind -> [dst]; dst -> {(src, kind)}.
        self._fwd = {}
        self._rev = {}
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

    def declare_graph_class(self, name, *, desc=None, parent=None, protocol=None):
        """Declare a graph class this provider owns; returns the qualified type
        for a Node subclass to set as ``type``::

            class DocTable(Table):
                type = irds.declare_graph_class('DocTable', parent=TABLE)

        ``parent`` is another qualified type -- this provider's own, or
        another provider's entirely (a third-party ``ext:MyTable`` may set
        ``parent='irds:Table'``) -- making this one a subtype of it for
        ``is_subclass``/``Graph.list(type=...)`` purposes. The type itself is
        still owned (and only ever registerable) by this provider; ``parent``
        is a graph relationship, not a transfer of ownership.

        ``protocol`` is the structural interface (see ``protocols.py``) nodes
        of this type satisfy: the Protocol class itself (stored as
        ``'module:Qualname'``) or that string, for a protocol that isn't
        importable here. A type without one inherits its parent's.
        """
        if protocol is not None and not isinstance(protocol, str):
            protocol = f'{protocol.__module__}:{protocol.__qualname__}'
        qualified = self.qualify(name)
        if not self.owns(qualified):
            raise ValueError(f'{self} cannot declare the type {name!r}')
        self.graph_classes[qualified] = {'desc': desc, 'parent': parent, 'protocol': protocol}
        return declare_graph_class(qualified, owner=self.prefix, desc=desc, parent=parent, protocol=protocol)

    def declare_graph_classes(self, module):
        """Declare every graph class a module's protocols define: each protocol
        class in ``module`` marked ``@protocols.graph_class`` becomes a type owned
        by this provider -- named after the class, its parent the type of the
        nearest marked base protocol, described by the first paragraph of its
        docstring, and with the class itself as its protocol. Returns
        ``{protocol class: qualified type}``, in definition order.

        So the protocol is the single declaration of a type::

            @graph_class
            @runtime_checkable
            class DocTable(Table, Protocol):
                \"\"\"A Table representing a collection of documents.\"\"\"

        A third party does the same with its own module and provider, with
        protocols that extend ours (``class MyTable(Table, Protocol)``).
        """
        import inspect
        out = {}
        for cls in vars(module).values():
            if not (isinstance(cls, type) and protocols.is_graph_class(cls)
                    and cls.__module__ == module.__name__):
                continue
            parent = None
            for base in cls.__bases__:
                if protocols.is_graph_class(base):
                    parent = out.get(base) or type_for_protocol(f'{base.__module__}:{base.__qualname__}')
                    break
            doc = inspect.getdoc(cls) or ''
            out[cls] = self.declare_graph_class(cls.__name__, desc=' '.join(doc.split('\n\n')[0].split()) or None,
                                      parent=parent, protocol=cls)
        return out

    def declare_graph_property(self, name, *, range='node', structural=False, desc=None):
        """Declare a graph property this provider owns (and return its qualified name).
        One declaration covers both kinds of property. An **edge** (the default,
        ``range='node'``) links a node to another; structural edges form the
        build DAG, informational ones are anything else. A **literal-valued
        property** has a ``range`` of ``'literal'`` (a value) or ``'json'`` (a JSON
        document). Any property may have several values on a node, as in RDF::

            irds.declare_graph_property('derivedFrom', structural=True, desc='...')
            irds.declare_graph_property('description', range='literal', desc='...')

        Names are lowerCamelCase, as RDF vocabularies write properties."""
        qualified = self.qualify(name)
        if not self.owns(qualified):
            raise ValueError(f'{self} cannot declare the graph property {name!r}')
        self.graph_properties[qualified] = {'range': range, 'structural': structural, 'desc': desc}
        return declare_graph_property(qualified, owner=self.prefix, range=range,
                                 structural=structural, desc=desc)

    @property
    def edge_kinds(self):
        """This provider's graph properties that are edges between nodes."""
        return {k: v for k, v in self.graph_properties.items() if v['range'] == 'node'}

    @property
    def literal_properties(self):
        """This provider's graph properties that carry a literal or JSON value."""
        return {k: v for k, v in self.graph_properties.items() if v['range'] in ('literal', 'json')}

    # -- registration -------------------------------------------------------

    def register(self, *nodes, module=None):
        """Add nodes to this provider.

        Stamps ``qualified_name`` (prefix applied), ``provider`` and
        ``defined_in`` (the calling module, or ``module=``). Ingests each node's
        ``structural_edges()``; a dependency given as an *object* that is not yet
        registered is registered too, transitively -- a registered node with an
        unregistered dependency would be an invalid graph, so the module need
        only register its roots. Dependencies given as *names* are assumed to be
        provided elsewhere. A bare name or edge
        kind means this provider's own namespace -- the one shorthand, and only
        inside a provider's own modules, where the provider is in plain sight.
        """
        module = module or _caller_module()
        for node in nodes:
            self._register_one(node, module)
        return nodes

    def _register_one(self, node, module):
        # What a node must have is whatever the Node protocol says, minus what
        # this registration assigns -- not a list kept in step with it by hand.
        missing = sorted(m for m in protocols.members(protocols.Node)
                         if m not in protocols.ASSIGNED_AT_REGISTRATION and not hasattr(node, m))
        if missing:
            raise TypeError(
                f'{node!r} does not look like a Node: missing {", ".join(missing)}. '
                f'A node needs no particular base class (see protocols.Node) -- '
                f'just this shape.')
        if node.provider is self and node.qualified_name in self.nodes:
            return
        if node.provider is not None and node.provider is not self:
            raise ValueError(
                f'{node.name!r} is already registered with {node.provider}')
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
        # After provider is set: a derived Benchmark's types depend on its
        # parent's facets, resolved through this provider's namespace.
        for graph_class in types_of(node):
            check_graph_class(graph_class)
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
        check_graph_class(generator.graph_class)
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
        node -- they are this provider's contribution."""
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
        """(forward, reverse) over this provider's edges."""
        self._bootstrap()
        fwd, rev = {}, {}
        for src, kinds in self._fwd.items():
            for kind, dsts in kinds.items():
                bucket = fwd.setdefault(src, {}).setdefault(kind, [])
                bucket.extend(d for d in dsts if d not in bucket)
        for dst, pairs in self._rev.items():
            rev.setdefault(dst, set()).update(pairs)
        return fwd, rev

    # -- catalog ------------------------------------------------------------

    def names(self):
        self._bootstrap()
        return set(self.nodes)

    def type_of(self, name):
        try:
            return types_of(self[name])[0]
        except KeyError:
            return None

    # -- protocols.Provider ---------------------------------------------------

    def discover_edges(self):
        """This provider's whole current catalog, as RDF-ready
        ``(subject, predicate, object)`` triples (``object`` wrapped in
        ``base.Literal`` when it's a property value rather than another
        node's name -- see ``row_triples``) -- every node its modules
        register, plus every edge it contributes. Never triggers a generator
        expansion (a parametric family is a rule, not a catalog entry).
        Imports the package's modules first. A subclass with a genuinely
        live catalog (``hf_provider.HfProvider``) overrides this entirely.
        """
        from .vocabulary import row_for
        self.import_all()
        for name, node in self.nodes.items():
            if name in self.generated:
                continue
            # A node needs no particular base class (see protocols.Node), so
            # discovery_literals is optional.
            literals = getattr(node, 'discovery_literals', lambda: {})()
            row = {**row_for(node), **{k: v for k, v in literals.items() if v}}
            yield from row_triples(name, row)
        fwd, _ = self.indexes()
        for src, kinds in fwd.items():
            for kind, dsts in kinds.items():
                for dst in dsts:
                    yield src, kind, dst
        for qualified, entry in self.graph_classes.items():
            if self.owns(qualified) and entry.get('parent'):
                yield qualified, RDFS_SUBCLASS_OF, entry['parent']
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
                        yield name, RDF_TYPE, generator.graph_class
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
        # Not registered yet: import every module once and retry.
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
        """Import every module in this provider's package."""
        if not self.package:
            return
        module = importlib.import_module(self.package)
        for info in pkgutil.iter_modules(module.__path__):
            importlib.import_module(f'{self.package}.{info.name}')

    def _bootstrap(self):
        if self._bootstrapped or not self.package:
            return False
        self._bootstrapped = True
        self.import_all()
        return True
