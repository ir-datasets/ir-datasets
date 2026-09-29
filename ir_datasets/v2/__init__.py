"""ir_datasets v2 — a knowledge graph of named, typed dataset nodes.

This package is a working sketch, not a replacement: it layers a declarative
node model over the existing v1 machinery (``ir_datasets.util`` downloads,
``ir_datasets.formats`` parsers, ``ir_datasets.indices`` docstores), so the
ideas can be exercised on real data without reimplementing any of it.

The generic graph machinery -- ``Node``, ``Edge``, ``Provider``, ``Graph``,
``Generator``, ``freeze``/``verify`` -- lives in this package (``base.py``,
``protocols.py``, ``registry.py``, ``graph.py``, ``vocabulary.py``,
``context.py``) but declares no vocabulary of its own; a *provider* declares
vocabulary, and any package (this one included) may be one. ``Provider``
(``protocols.py``) is a two-method contract -- resolve a node by name, report
the provider's whole catalog as edges -- not a class to inherit; a provider
is free to keep a list of registered datasets internally however it likes.
``ManifestProvider`` (``registry.py``) is the batteries-included
implementation of it -- registration, vocabulary declaration, generators, a
frozen manifest -- and what ``irds`` (this package's own provider) is built
from. ``ir_datasets.v2`` contributes ten node types, all owned by its own
``irds`` provider:

* ``Resource`` — bytes, and where to get them
* ``Table`` — a structured set of records, parsed from a source -- and its six
  per-entity subtypes, ``DocTable``, ``QueryTable``, ``QrelTable``,
  ``RunTable``, ``DocPairTable``, ``QlogTable`` (and their format subclasses).
  Each is its own declared graph type (``is_subtype('irds:QrelTable',
  'irds:Table')`` is true), so ``list_datasets(type='irds:Table')`` finds all
  six kinds while ``list_datasets(type='irds:QrelTable')`` finds only qrels
  tables. ``QlogTable`` (raw query logs, e.g. AOL's/TripClick's session data)
  is the one entity that is never a Benchmark facet -- see ``nodes.py``'s
  ``ENTITIES``/``TABLE_TYPES`` docstrings for why.
* ``Benchmark`` — docs + queries + qrels (etc.) bundled into an evaluable task,
  plus flat metadata (``citation``, ``metrics``)
* ``Suite`` — a named, structural set of related Benchmarks (e.g. BEIR)

Citations-as-papers and metrics-as-measures (each with their own edges) are
future work for a broader knowledge graph; for now both are plain metadata.

The design still decentralizes: a third-party package can declare its own
``Provider``, its own node types and edge kinds, and its own entry point in the
``ir_datasets.providers`` group (``ENTRY_POINT_GROUP``) -- pointing at nodes
this package's ``irds`` provider owns, or vice versa -- without either package
importing the other until a node is actually resolved. A type it declares may
also be a *subtype* of one of ours (``ext.node_type('MyTable', parent=TABLE)``),
so a new entity kind this package never enumerated still shows up under
``list(type='irds:Table')``.

Nothing about being a node requires inheriting from this package's classes,
either. ``Resource``/``Table``/``Benchmark``/``Suite`` (and their common shape,
``Node``) are structural contracts (see ``protocols.py``), not base classes --
``ResourceProtocol``/``TableProtocol``/``BenchmarkProtocol``/``SuiteProtocol``/
``NodeProtocol`` below are the actual interfaces; the classes of the same name
without the suffix are one convenient, inheritable implementation of them, not
a requirement. A third-party ``Table`` backed by a live API instead of a
downloaded ``Resource`` is exactly as valid a graph node as any of ours.

Every name has a home and the home is in the name: ``irds:antique-test``,
type ``irds:Benchmark``, edge kind ``irds:derived_from``. There is no default
provider, beyond ``load()``'s own bare-name fallback to ``irds:``/``legacy:``
(see ``graph.py``'s docstring); legacy v1 ids (``antique/test``) still
resolve that way, to whatever v2 node replaced them.

The default API is property access (what v1 called the "beta" API), with the
legacy method API available on the same objects::

    ds = ir_datasets.v2.load('irds:antique-test')

    len(ds.docs), len(ds.queries), len(ds.qrels)   # counts (free if frozen)
    ds.queries[0], ds.queries[-1], ds.docs[:10]    # index / slice
    ds.docs.lookup(['2020338_0'])                  # random access
    ds.docs.record_type, ds.qrels.defs             # schema / relevance levels

    ds.docs_iter(), ds.docs_count(), ds.docs_store()   # legacy, still works
    ds.has_qrels(), ds.queries_handler()

Every table is also loadable on its own::

    len(ir_datasets.v2.load('irds:antique-test-queries'))
"""
from .base import Edge, Generator, Literal, Node, Param
from .registry import DuplicateNameError, ManifestProvider
from .graph import ENTRY_POINT_GROUP, Graph, default_graph, discover
from .vocabulary import (
    edge_kinds, informational_kinds, is_structural, is_subtype, node_types,
    structural_kinds,
)

from .provider import irds
from .hf_provider import hf
from .clirmatrix_provider import clirmatrix
# clirmatrix is generator-only (no frozen node rows of its own to trigger a
# lazy per-name import the way every hand-written irds module does -- see
# registry.ManifestProvider.__getitem__'s `row = self.frozen(name)` path) --
# so, like hf above, its module needs importing here to register its
# Generators at all, rather than only on first (already-too-late) lookup.
from .datasets import clirmatrix as _clirmatrix  # noqa: F401
# legacy is also generator-only for its CLIRMatrix share -- see
# legacy_provider.py's own docstring.
from .legacy_provider import legacy
from .nodes import (
    BENCHMARK, Benchmark, DEFAULTABLE, Directory, DocPairTable, DocTable, ENTITIES,
    File, GitRepo, QlogTable, RESOURCE, Resource,
    QrelTable, QueryTable, STRUCTURAL_EDGES, SUITE,
    SUITE_MEMBER, RunTable, Suite, TABLE, TABLE_TYPES, Table, source_resources,
)
from .formats import (
    Parser, TrecDocs, TrecQrels, TrecQueries, TrecScoredDocs, TsvDocPairs,
    TsvDocs, TsvQueries,
)
from .sources import Readable, Source, external_home, transform
from .filters import DerivedTable, Filter, ids_from_lines, ids_of
from .protocols import Node as NodeProtocol
from .protocols import Resource as ResourceProtocol
from .protocols import Table as TableProtocol
from .protocols import Benchmark as BenchmarkProtocol
from .protocols import Suite as SuiteProtocol
#: The minimal contract a provider must satisfy to join a Graph -- see
#: ``protocols.Provider``'s own docstring. ``ManifestProvider`` above is one
#: (big) implementation of it, not the contract itself.
from .protocols import Provider

# Subscribe: our providers join the default graph. (Also declared as entry
# points, which is how an installed package is found without being imported.)
graph = default_graph()
graph.add(irds)
graph.add(hf)
graph.add(clirmatrix)
graph.add(legacy)


def load(name):
    """Resolve a node by qualified name (``irds:antique-test``), or by legacy
    v1 id (``antique/test``)."""
    return graph[name]


def list_datasets(type=None):
    """Qualified node names; ``type`` is a qualified type (``irds:Benchmark``).
    Always each provider's full, current catalog -- for a provider whose
    discovery is live (``hf``, crawling the Hub) that crawl runs to build it,
    once per process (``graph`` caches it; see ``Graph.list``)."""
    return graph.list(type=type)


def external_status():
    """Every Resource the user must supply themselves (``Source.external``), as
    ``(name, path, present, instructions)`` -- where to put it under
    ``external_home()`` (``<home>/external`` or ``$IR_DATASETS_EXTERNAL``), and
    whether it's there yet."""
    from .sources import _ManualSource
    rows = []
    for name in list_datasets(type=RESOURCE):
        try:
            node = graph[name]
        except Exception:
            continue
        for src in getattr(node, 'sources', ()):
            if isinstance(src, _ManualSource):
                # (name, where it is now or belongs, present?, instructions)
                path = src.local_path
                rows.append((name, str(path), path.exists(),
                             (src.instructions or '').format(path=src.default_path)))
    return rows


def citation(name):
    """The node's citation, or None. Plain metadata for now -- see the module
    docstring."""
    return graph[name].metadata.get('citation')


__all__ = [
    # generic graph machinery
    'Node', 'Edge', 'Literal', 'Generator', 'Param', 'Provider', 'ManifestProvider', 'Graph',
    'DuplicateNameError', 'default_graph', 'discover', 'ENTRY_POINT_GROUP',
    'node_types', 'edge_kinds', 'structural_kinds', 'informational_kinds', 'is_structural',
    'is_subtype',
    # nodes
    'Resource', 'File', 'Directory', 'GitRepo', 'Table', 'DocTable', 'QueryTable',
    'QrelTable', 'RunTable', 'DocPairTable', 'QlogTable', 'Benchmark', 'Suite', 'ENTITIES',
    # protocols: the structural contracts Resource/Table/Benchmark/Suite/Node above
    # are one implementation of, not requirements to inherit from
    'NodeProtocol', 'ResourceProtocol', 'TableProtocol', 'BenchmarkProtocol', 'SuiteProtocol',
    # vocabulary (declared on the irds provider)
    'RESOURCE', 'TABLE', 'TABLE_TYPES', 'BENCHMARK', 'SUITE', 'SUITE_MEMBER',
    'STRUCTURAL_EDGES', 'DEFAULTABLE', 'source_resources',
    # sources
    'Source', 'Readable', 'transform', 'external_home',
    # formats (the node type IS the format)
    'TsvDocs', 'TsvQueries', 'TrecDocs', 'TrecQueries', 'TrecQrels', 'TrecScoredDocs', 'TsvDocPairs',
    'Parser',
    # derivation
    'Filter', 'DerivedTable', 'ids_of', 'ids_from_lines',
    # providers and the graph
    'irds', 'hf', 'clirmatrix', 'legacy', 'graph',
    'load', 'list_datasets', 'citation', 'external_status',
]
