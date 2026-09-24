"""ir_datasets v2 — a knowledge graph of named, typed dataset nodes.

This package is a working sketch, not a replacement: it layers a declarative
node model over the existing v1 machinery (``ir_datasets.util`` downloads,
``ir_datasets.formats`` parsers, ``ir_datasets.indices`` docstores), so the
ideas can be exercised on real data without reimplementing any of it.

The generic graph machinery -- ``Node``, ``Edge``, ``Provider``, ``Graph``,
``Generator``, ``freeze``/``verify`` -- lives in this package (``base.py``,
``registry.py``, ``graph.py``, ``vocabulary.py``, ``context.py``) but declares
no vocabulary of its own; a *provider* declares vocabulary, and any package
(this one included) may be one. ``ir_datasets.v2`` contributes exactly four
node types, all owned by its own ``irds`` provider:

* ``Resource`` — bytes, and where to get them
* ``Table`` (``Docs``/``Queries``/``Qrels``/``ScoredDocs``/``DocPairs`` and their
  format subclasses) — a structured set of records, parsed from a source
* ``Benchmark`` — docs + queries + qrels (etc.) bundled into an evaluable task,
  plus flat metadata (``citation``, ``metrics``)
* ``Suite`` — a named, structural set of related Benchmarks (e.g. BEIR)

Citations-as-papers and metrics-as-measures (each with their own edges) are
future work for a broader knowledge graph; for now both are plain metadata.

The design still decentralizes: a third-party package can declare its own
``Provider``, its own node types and edge kinds, and its own entry point in the
``ir_datasets.providers`` group (``ENTRY_POINT_GROUP``) -- pointing at nodes
this package's ``irds`` provider owns, or vice versa -- without either package
importing the other until a node is actually resolved.

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
provider; legacy v1 ids (``antique/test``) still resolve, as explicit aliases.

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
from .base import Edge, Generator, Node, Param
from .registry import DuplicateNameError, Provider
from .graph import ENTRY_POINT_GROUP, Graph, default_graph, discover
from .vocabulary import (
    edge_kinds, informational_kinds, is_structural, node_types, structural_kinds,
)

from .provider import irds
from .hf_provider import hf
from .nodes import (
    BENCHMARK, Benchmark, DEFAULTABLE, Directory, DocPairs, Docs, ENTITIES, File,
    GitRepo, RESOURCE, Resource,
    INFORMATIONAL_EDGES, Qrels, Queries, STRUCTURAL_EDGES, SUITE,
    SUITE_BENCHMARK, ScoredDocs, Suite, TABLE, Table, source_resources,
)
from .formats import (
    Parser, TrecDocs, TrecQrels, TrecQueries, TrecScoredDocs, TsvDocPairs,
    TsvDocs, TsvQueries,
)
from .sources import Readable, Source, transform
from .filters import DerivedTable, Filter, ids_from_lines, ids_of
from .protocols import Node as NodeProtocol
from .protocols import Resource as ResourceProtocol
from .protocols import Table as TableProtocol
from .protocols import Benchmark as BenchmarkProtocol
from .protocols import Suite as SuiteProtocol

# Subscribe: our providers join the default graph. (Also declared as entry
# points, which is how an installed package is found without being imported.)
graph = default_graph()
graph.add(irds)
graph.add(hf)


def load(name):
    """Resolve a node by qualified name (``irds:antique-test``), or by legacy
    v1 id (``antique/test``)."""
    return graph[name]


def list_datasets(type=None, discover=False):
    """Qualified node names; ``type`` is a qualified type (``irds:Benchmark``).
    ``discover=True`` also asks providers with nothing to bootstrap/enumerate
    locally (e.g. ``hf``) to report their known names via a live lookup."""
    return graph.list(type=type, discover=discover)


def citation(name):
    """The node's citation, or None. Plain metadata for now -- see the module
    docstring."""
    return graph[name].metadata.get('citation')


__all__ = [
    # generic graph machinery
    'Node', 'Edge', 'Generator', 'Param', 'Provider', 'Graph', 'DuplicateNameError',
    'default_graph', 'discover', 'ENTRY_POINT_GROUP',
    'node_types', 'edge_kinds', 'structural_kinds', 'informational_kinds', 'is_structural',
    # nodes
    'Resource', 'File', 'Directory', 'GitRepo', 'Table', 'Docs', 'Queries', 'Qrels',
    'ScoredDocs', 'DocPairs', 'Benchmark', 'Suite', 'ENTITIES',
    # protocols: the structural contracts Resource/Table/Benchmark/Suite/Node above
    # are one implementation of, not requirements to inherit from
    'NodeProtocol', 'ResourceProtocol', 'TableProtocol', 'BenchmarkProtocol', 'SuiteProtocol',
    # vocabulary (declared on the irds provider)
    'RESOURCE', 'TABLE', 'BENCHMARK', 'SUITE', 'SUITE_BENCHMARK',
    'STRUCTURAL_EDGES', 'INFORMATIONAL_EDGES', 'DEFAULTABLE', 'source_resources',
    # sources
    'Source', 'Readable', 'transform',
    # formats (the node type IS the format)
    'TsvDocs', 'TsvQueries', 'TrecDocs', 'TrecQueries', 'TrecQrels', 'TrecScoredDocs', 'TsvDocPairs',
    'Parser',
    # derivation
    'Filter', 'DerivedTable', 'ids_of', 'ids_from_lines',
    # providers and the graph
    'irds', 'hf', 'graph',
    'load', 'list_datasets', 'citation',
]
