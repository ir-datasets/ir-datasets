"""The structural interfaces a graph node satisfies -- not base classes.

Nothing here requires inheritance. A ``Node`` is any object with these
attributes and methods; ``registry.Provider.register()`` and ``graph.Graph``
only ever touch a node through this shape, never through ``isinstance``
against a concrete class. That's deliberate: the concrete classes in
``base.py``/``nodes.py`` (``base.Node``, ``nodes.Resource``, ``nodes.Table``,
``nodes.Benchmark``, ``nodes.Suite``) are *one* implementation of these
protocols -- a convenient one to inherit from, so authors don't re-write
``__init__``'s bookkeeping by hand -- but they are not the contract. A
third-party package can build a node -- a ``Table`` backed by a live API
instead of a downloaded ``Resource``, say -- that satisfies ``Table`` below
without importing anything from this package except (optionally) this module,
and it joins the graph exactly as any of ours does.

Protocols may declare attributes as well as methods (PEP 544) -- a bare
annotation (``name: str``) is satisfied by an instance attribute set however
the implementer likes, not only by a matching method. ``Node``'s
``qualified_name``/``provider``/``defined_in`` are exactly that: plain
settable attributes, because ``Provider.register()`` assigns them directly
(``node.qualified_name = name``).

These are marked ``@runtime_checkable`` for convenience (``isinstance(x, Table)``
reads well in a test), but that check is presence-only: it confirms the named
attributes and methods exist, never their types or signatures. Nothing in this
package relies on it for anything load-bearing -- see ``registry.py``'s own,
narrower check.
"""
from typing import Any, ContextManager, Optional, Protocol, runtime_checkable


@runtime_checkable
class Node(Protocol):
    """What ``Provider.register()`` and ``Graph`` require of every node."""
    name: str
    type: str
    metadata: dict
    #: Assigned by the provider at registration; ``None`` before that.
    qualified_name: Optional[str]
    provider: Optional[Any]
    defined_in: Optional[str]

    def structural_edges(self) -> list:
        """This node's dependencies, as Edges (object or name targets)."""
        ...

    def attest(self, *, verify: bool = False, **options) -> Optional[dict]:
        """Data ``freeze`` should record for this node, or None."""
        ...

    def verify(self, frozen: dict, **options) -> list:
        """Divergences between this node and its frozen row."""
        ...


@runtime_checkable
class Resource(Node, Protocol):
    """Bytes, and where to get them."""
    sources: list
    size: Optional[int]
    cache_path: Any

    def path(self, force: bool = True) -> str: ...
    def stream(self) -> ContextManager: ...


@runtime_checkable
class Table(Node, Protocol):
    """A homogeneous set of records of one ``entity`` kind."""
    entity: str

    def count(self) -> int: ...
    @property
    def record_type(self) -> type: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __getitem__(self, key): ...
    def lookup(self, ids): ...


@runtime_checkable
class Benchmark(Node, Protocol):
    """Facets (docs/queries/qrels/...) bundled into one evaluable task."""
    def edge(self, entity: str) -> Optional[Table]: ...
    def has(self, entity: str) -> bool: ...


@runtime_checkable
class Suite(Node, Protocol):
    """A named set of related Benchmarks."""
    benchmarks: list


@runtime_checkable
class Provider(Protocol):
    """What joins a package to the graph. ``Graph`` touches a provider through
    exactly these two members -- nothing else is load-bearing.

    ``registry.ManifestProvider`` is one implementation of this (a big one:
    registration, vocabulary declaration, a frozen manifest, generators) --
    a convenient default for a provider with a static catalog to freeze, not
    the contract itself. A provider can be as small as a ``name``, a
    ``load()``, and a ``discover_edges()`` that yields nothing; a typical one
    keeps a list of registered datasets internally, however it likes.
    """
    #: The namespace prefix this provider owns (``irds``, ``hf``, ...).
    #: Every node it resolves is named ``{prefix}:...``.
    prefix: str

    def load(self, name: str) -> Node:
        """Resolve one (already-qualified) name to a node. KeyError if this
        provider has nothing by that name."""
        ...

    def discover_edges(self):
        """This provider's whole current catalog, as RDF-ready
        ``(subject, predicate, object)`` rows -- ``object`` wrapped in
        ``base.Literal`` when it's a property value rather than another
        node's name (see ``registry.row_triples``) -- a ``type`` row per node
        plus one row per field/edge, the same shape a
        manifest export uses. Never materializes a real node (``load()`` does
        that); this is the cheap-to-read side of the graph -- listing,
        traversal, validation -- that works without importing or downloading
        anything a node itself would need.

        Always does full discovery -- a live crawl, if that's what finding
        everything takes (see ``hf``, which lists the Hub here). There is no
        separate cheap/expensive mode: a provider that wants a fast default
        should cache its own result and invalidate it on its own terms;
        ``Graph`` caches its own call to this per provider so a query doesn't
        repeat the work, but only for that ``Graph``'s lifetime.
        """
        ...
