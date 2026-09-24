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
