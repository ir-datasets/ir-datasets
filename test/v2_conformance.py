"""Generic conformance harness for v2 nodes.

In v1 every family needed its own integration test with hand-pasted counts and
sample records (test/integration/antique.py and ~60 siblings). Here there is
structural tests: they walk every registered node and check it. Adding a
dataset adds no test code.

    python -m unittest test.v2_conformance               # everything
    python -m unittest test.v2_conformance.TestV2Graph    # structure only (no download)
"""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ir_datasets.util
import ir_datasets.v2 as v2
from ir_datasets.v2 import graph, irds
from ir_datasets.v2 import sources as sourcesmod
from ir_datasets.v2.sources import local_copy_hint


def _scratch(prefix='test'):
    """An isolated provider + graph; nothing here touches the real ones."""
    src = v2.Resource('scratch-file', sources=['https://example.org/x.tsv'])
    table = v2.TsvQueries('scratch-queries', source=src)
    provider = v2.ManifestProvider(prefix)
    return provider, src, table


class TestV2Graph(unittest.TestCase):
    """Structural checks over every registered node: no data is downloaded."""

    def test_catalog_is_not_empty(self):
        self.assertTrue(irds.names(), 'no nodes registered')

    def test_every_node_resolves(self):
        for name in sorted(irds.names()):
            with self.subTest(node=name):
                self.assertIsNotNone(graph[name])

    def test_graph_classes_are_known(self):
        known = {v2.FILE, v2.DIRECTORY, v2.BENCHMARK, v2.ADHOC_BENCHMARK, v2.QA_BENCHMARK,
                 v2.SUITE, v2.BUNDLE, v2.OVERVIEW, *v2.TABLE_TYPES.values()}
        for name in sorted(irds.names()):
            with self.subTest(node=name):
                self.assertTrue(set(graph.types_of(name)) <= known)
                self.assertIn(graph.type_of(name), known)

    def test_exactly_these_graph_classes(self):
        """The abstract parents (Resource, Table, Bundle), Alias (a stand-in for another node; legacy:V1Dataset is one), the leaf types
        under each (File/Directory; one per table entity, incl. QlogTable,
        which is never a Benchmark facet -- see nodes.py's
        ENTITIES/TABLE_TYPES docstrings; Benchmark/Suite, with the
        task-specific Adhoc/Qa benchmark types under Benchmark). No more."""
        self.assertEqual({v2.DATA, v2.FILE, v2.DIRECTORY, v2.TABLE, *v2.TABLE_TYPES.values(),
                          v2.BUNDLE, v2.OVERVIEW, v2.BENCHMARK, v2.ADHOC_BENCHMARK, v2.QA_BENCHMARK, v2.SUITE, v2.ALIAS},
                         set(irds.graph_classes))

    def test_type_hierarchy(self):
        for sub, parent in [(v2.FILE, v2.DATA), (v2.DIRECTORY, v2.DATA),
                            (v2.BENCHMARK, v2.BUNDLE), (v2.SUITE, v2.BUNDLE), (v2.OVERVIEW, v2.BUNDLE),
                            (v2.ADHOC_BENCHMARK, v2.BENCHMARK), (v2.QA_BENCHMARK, v2.BUNDLE)]:
            self.assertTrue(v2.is_subclass(sub, parent), (sub, parent))
        self.assertFalse(v2.is_subclass(v2.TABLE, v2.BUNDLE))
        # legacy declares its own type under irds:Alias (cross-provider parent)
        self.assertTrue(v2.is_subclass('legacy:V1Dataset', v2.ALIAS))
        self.assertFalse(v2.is_subclass('legacy:V1Dataset', v2.DATA))

    def test_declared_protocols_resolve_and_nodes_satisfy_them(self):
        """Every type's protocol (its own or inherited) names an importable
        Protocol class, and a real node of each type satisfies it."""
        import importlib
        from ir_datasets.v2.vocabulary import protocol_of

        def resolve(ref):
            module, _, qualname = ref.partition(':')
            return getattr(importlib.import_module(module), qualname)

        for t in {v2.DATA, v2.FILE, v2.DIRECTORY, v2.TABLE, *v2.TABLE_TYPES.values(),
                  v2.BUNDLE, v2.BENCHMARK, v2.ADHOC_BENCHMARK, v2.QA_BENCHMARK, v2.SUITE,
                  v2.ALIAS, 'legacy:V1Dataset'}:
            with self.subTest(type=t):
                ref, declared_on = protocol_of(t)
                self.assertIsNotNone(ref, t)
                protocol = resolve(ref)
                names = v2.list_datasets(type=t)
                if not names:
                    continue  # an abstract parent with no nodes of its own
                provider = v2.graph.provider_for(names[0])
                self.assertIsInstance(provider.load(names[0]), protocol, names[0])

    def test_type_parents_mirror_protocol_bases(self):
        """A type's parent is the type of its protocol's marked base protocol:
        the hierarchy is written once, in protocols.py."""
        import importlib
        from ir_datasets.v2 import protocols
        for t, info in v2.graph_classes().items():
            if info['owner'] != 'irds' or not info.get('protocol'):
                continue
            module, _, qualname = info['protocol'].partition(':')
            cls = getattr(importlib.import_module(module), qualname)
            marked_bases = [b for b in cls.__bases__ if protocols.is_graph_class(b)]
            parent = info['parent']
            with self.subTest(type=t):
                if marked_bases:
                    self.assertEqual(f'{marked_bases[0].__module__}:{marked_bases[0].__qualname__}',
                                     v2.graph_classes()[parent]['protocol'])
                else:
                    self.assertIsNone(parent)
                self.assertEqual(' '.join(cls.__doc__.split('\n\n')[0].split()), info['desc'])

    def test_a_provider_declares_types_from_its_protocols(self):
        import sys
        import types
        from typing import Protocol, runtime_checkable
        from ir_datasets.v2 import protocols
        module = types.ModuleType('acme_protocols')

        @protocols.graph_class
        @runtime_checkable
        class ThingTable(protocols.Table, Protocol):
            """A table of things.

            More detail that is not part of the short description."""
        ThingTable.__module__ = module.__name__
        module.ThingTable = ThingTable
        acme = v2.ManifestProvider('acmedecl')
        declared = acme.declare_graph_classes(module)
        self.assertEqual({ThingTable: 'acmedecl:ThingTable'}, declared)
        info = v2.graph_classes()['acmedecl:ThingTable']
        self.assertEqual('irds:Table', info['parent'])
        self.assertEqual('A table of things.', info['desc'])
        self.assertTrue(v2.is_subclass('acmedecl:ThingTable', 'irds:Table'))

    def test_protocol_class_or_string_is_stored_as_module_name(self):
        from ir_datasets.v2.protocols import Table
        from ir_datasets.v2.registry import ManifestProvider
        acme = ManifestProvider('acmeproto')
        by_class = acme.declare_graph_class('ByClass', protocol=Table)
        by_string = acme.declare_graph_class('ByString', protocol='acme.protocols:Thing')
        self.assertEqual('ir_datasets.v2.protocols:Table', v2.graph_classes()[by_class]['protocol'])
        self.assertEqual('acme.protocols:Thing', v2.graph_classes()[by_string]['protocol'])

    def test_benchmark_subtypes_follow_their_tables(self):
        """AdhocBenchmark needs docs + queries + qrels; QaBenchmark needs
        queries + answers (docs optional); one with both is both."""
        for name in sorted(irds.names()):
            if not v2.is_subclass(graph.type_of(name), v2.BENCHMARK):
                continue
            with self.subTest(node=name):
                node, types = graph[name], set(graph.types_of(name))
                self.assertEqual(
                    node.has('docs') and node.has('queries') and node.has('qrels'),
                    v2.ADHOC_BENCHMARK in types)
                self.assertEqual(node.has('queries') and node.has('answers'),
                                 v2.QA_BENCHMARK in types)
        both = v2.list_datasets(type=v2.ADHOC_BENCHMARK)
        self.assertTrue(set(both) & set(v2.list_datasets(type=v2.QA_BENCHMARK)))
        self.assertLessEqual(set(v2.list_datasets(type=v2.BENCHMARK)) | set(v2.list_datasets(type=v2.SUITE)),
                             set(v2.list_datasets(type=v2.BUNDLE)))

    def test_benchmark_edges_resolve(self):
        for name in sorted(irds.names()):
            node = graph[name]
            if isinstance(node, v2.Benchmark):
                with self.subTest(node=name):
                    self.assertTrue(any(node.has(e) for e in v2.ENTITIES))

    def test_no_hidden_tier(self):
        """Every registered node is listed -- no leading-underscore convention
        (dropped; it was a listing rule with no model backing, not worth the
        two-tier mental model)."""
        antique_resource_names = [n for n in v2.list_datasets(type=v2.DATA)
                                  if 'antique' in n]
        self.assertTrue(antique_resource_names)
        for name in antique_resource_names:
            self.assertIsNotNone(graph[name])

    def test_legacy_ids_resolve_to_their_v2_replacement(self):
        """graph[v1_id] loads legacy:v1_id, sees a legacy:V1Dataset, and
        follows its alias_of edge -- see graph.py's own docstring."""
        from ir_datasets.v2.legacy_provider import _REPLACED_BY
        self.assertTrue(_REPLACED_BY)
        for v1_id, target in _REPLACED_BY.items():
            with self.subTest(v1_id=v1_id):
                self.assertEqual(target, graph[v1_id].qualified_name)

    def test_legacy_id_without_replacement_raises(self):
        from ir_datasets.v2.legacy_provider import _REPLACED_BY, _V1_IDS
        orphan = next(v1_id for v1_id in _V1_IDS if v1_id not in _REPLACED_BY)
        with self.assertRaisesRegex(KeyError, 'no known target'):
            graph[orphan]

    def test_bare_names_fall_back_to_irds_then_legacy(self):
        """No general default provider -- but load() tries irds: first, then
        legacy:, for a bare name (see graph.py's docstring)."""
        node = graph['irds:antique-test']
        self.assertEqual('irds:antique-test', node.qualified_name)
        self.assertIs(node, graph['antique-test'])       # bare -> irds: fallback
        self.assertIs(node, graph['antique/test'])        # bare -> legacy: fallback, then replaced_by
        with self.assertRaisesRegex(KeyError, 'not a qualified name'):
            graph['totally-bogus-name-that-does-not-exist-anywhere']
        self.assertTrue(all(':' in n for n in v2.list_datasets()))

    def test_vocabulary_is_owned_and_qualified(self):
        self.assertEqual('irds:AdhocBenchmark', graph['irds:antique-test'].type)
        self.assertEqual('irds:Docs', v2.TsvDocs.type)
        self.assertEqual('irds:Qrels', v2.TrecQrels.type)
        self.assertEqual('irds', v2.graph_classes()['irds:Docs']['owner'])
        self.assertEqual('irds', v2.edge_kinds()['irds:derivedFrom']['owner'])
        docs = v2.list_datasets(type='irds:Docs')
        self.assertIn('irds:antique-docs', docs)
        self.assertTrue(all(v2.graph.type_of(n) == 'irds:Docs' for n in docs))
        qrels = v2.list_datasets(type='irds:Qrels')
        self.assertIn('irds:antique-test-qrels', qrels)
        self.assertTrue(all(v2.graph.type_of(n) == 'irds:Qrels' for n in qrels))
        # the provider records its own vocabulary
        self.assertIn('irds:Docs', irds.graph_classes)
        self.assertIn('irds:Qrels', irds.graph_classes)
        self.assertTrue(irds.edge_kinds['irds:derivedFrom']['structural'])

    def test_table_types_are_subtypes_of_table(self):
        """Each per-entity type is a declared subtype of the generic
        irds:Table -- so a type-filtered listing can ask for either the
        specific kind or the whole family."""
        for entity, qualified in v2.TABLE_TYPES.items():
            with self.subTest(entity=entity):
                self.assertTrue(v2.is_subclass(qualified, 'irds:Table'))
                self.assertTrue(v2.is_subclass(qualified, qualified))
        self.assertFalse(v2.is_subclass('irds:Table', 'irds:Qrels'))
        self.assertFalse(v2.is_subclass('irds:Benchmark', 'irds:Table'))

        tables = v2.list_datasets(type='irds:Table')
        self.assertIn('irds:antique-docs', tables)         # a DocTable
        self.assertIn('irds:antique-test-qrels', tables)    # a QrelTable
        self.assertNotIn('irds:antique-test', tables)       # a Benchmark
        self.assertTrue(all(v2.is_subclass(v2.graph.type_of(n), 'irds:Table')
                            for n in tables))

    def test_undeclared_graph_class_cannot_register(self):
        class Rogue(v2.Node):
            type = 'acme:never-declared'
        with self.assertRaisesRegex(ValueError, 'undeclared graph class'):
            v2.ManifestProvider('acme').register(Rogue('x'))
        class Typeless(v2.Node):
            pass
        with self.assertRaises(ValueError):
            v2.ManifestProvider('acme').register(Typeless('x'))

    def test_provider_declares_only_in_its_own_namespace(self):
        acme = v2.ManifestProvider('acme')
        self.assertEqual('acme:thing', acme.declare_graph_class('thing'))
        self.assertEqual('acme:made_from', acme.declare_graph_property('made_from', structural=True))
        with self.assertRaises(ValueError):
            acme.declare_graph_class('irds:thing')
        with self.assertRaises(ValueError):
            acme.declare_graph_property('irds:derivedFrom', structural=True)

    def test_providers_subscribe_to_the_default_graph(self):
        self.assertEqual(['clirmatrix', 'documentation', 'hf', 'irds', 'legacy'], sorted(graph.providers))
        self.assertIs(v2.default_graph().providers, graph.providers)

    def test_minimal_provider_satisfies_the_protocol(self):
        """A provider need not be a ManifestProvider -- protocols.Provider is
        just `prefix`, `load(name)`, `discover_edges()`; no registration, no
        vocabulary. discover_edges() always runs its
        full discovery (there is no cheap/expensive flag to opt into)."""
        class MadeUp:
            prefix = 'scratchknown'

            def load(self, name):
                raise KeyError(name)

            def discover_edges(self):
                yield 'scratchknown:made-up-name', 'rdf:type', 'irds:Table'

        provider = MadeUp()
        self.assertIsInstance(provider, v2.Provider)
        scratch_graph = v2.Graph([provider])
        self.assertIn('scratchknown:made-up-name', scratch_graph.list())

    def test_entry_points_name_the_prefix(self):
        """pyproject declares each provider under the prefix it owns."""
        import tomllib
        with open(Path(__file__).parent.parent / 'pyproject.toml', 'rb') as f:
            eps = tomllib.load(f)['project']['entry-points'][v2.ENTRY_POINT_GROUP]
        for name, spec in eps.items():
            with self.subTest(entry_point=name):
                module, _, attr = spec.partition(':')
                self.assertEqual(name, getattr(importlib.import_module(module), attr).prefix)


class TestV2Registration(unittest.TestCase):
    """Construction is inert; registration is explicit and transitive."""

    def test_construction_does_not_register(self):
        _, src, table = _scratch()
        self.assertIsNone(table.provider)
        self.assertIsNone(table.qualified_name)
        self.assertNotIn('irds:scratch-queries', irds.nodes)
        # ...but the object is fully usable
        self.assertEqual([v2.Edge('irds:derivedFrom', src)], table.structural_edges())

    def test_register_applies_prefix_and_pulls_in_dependencies(self):
        provider, src, table = _scratch('acme')
        provider.register(table)
        self.assertEqual('acme:scratch-queries', table.qualified_name)
        self.assertIs(provider, table.provider)
        self.assertEqual(__name__, table.defined_in)
        # the Resource came along, and the edge was qualified
        self.assertEqual('acme:scratch-file', src.qualified_name)
        self.assertEqual({'irds:derivedFrom': ['acme:scratch-file']},   # acme uses irds' vocabulary
                         v2.Graph([provider]).edges_of('acme:scratch-queries'))

    def test_cannot_register_with_two_providers(self):
        provider, _, table = _scratch()
        provider.register(table)
        with self.assertRaises(ValueError):
            v2.ManifestProvider('other').register(table)

    def test_duplicate_name_from_different_module_is_an_error(self):
        provider, _, table = _scratch()
        provider.register(table)
        other = v2.TsvQueries('scratch-queries',
                              source=v2.Resource('f2', sources=['https://x/y']))
        with self.assertRaises(v2.DuplicateNameError):
            provider.register(other, module='somewhere.else')

    def test_registered_nodes_live_in_irds(self):
        node = graph['irds:antique-test']
        self.assertIs(irds, node.provider)
        self.assertEqual('ir_datasets.v2.datasets.antique', node.defined_in)
        # five roots registered; everything they need came along
        self.assertIn('irds:antique-docs.txt', irds.nodes)
        self.assertIn('irds:antique-test-qrels', irds.nodes)

    def test_derived_facets_register_with_their_benchmark(self):
        facet = graph['irds:antique-test-non-offensive'].queries
        self.assertIs(irds, facet.provider)
        self.assertEqual('ir_datasets.v2.datasets.antique', facet.defined_in)
        self.assertIs(facet, graph['irds:antique-test-non-offensive-queries'])
        # ...and the derived benchmark has facet edges of its own
        edges = graph.edges_of('irds:antique-test-non-offensive', structural_only=True)
        self.assertEqual(['irds:antique-test-non-offensive-queries'], edges['irds:queries'])
        self.assertEqual(['irds:antique-docs'], edges['irds:docs'])   # corpus shared by reference
        # order isn't semantically meaningful, so compare as sets
        self.assertCountEqual(['irds:antique-test-queries', 'irds:antique-test-disallow-list.txt'],
                              graph.edges_of(facet.qualified_name)['irds:derivedFrom'])

    def test_bare_names_and_kinds_mean_own_namespace(self):
        """The one shorthand, usable only inside a provider's own modules."""
        provider, src, table = _scratch('acme')
        provider.declare_graph_property('like', structural=False)
        provider.register(table)
        provider.add_edge('scratch-queries', 'like', 'other')
        self.assertEqual([['acme:scratch-queries', 'acme:like', 'acme:other']],
                         [r for r in provider.edge_rows() if r[1] == 'acme:like'])

    def test_names_may_not_contain_colons(self):
        with self.assertRaises(ValueError):
            v2.Resource('acme:file', sources=['https://x/y'])
        with self.assertRaises(ValueError):
            v2.ManifestProvider('')


class _FromScratchTable:
    """A Table-shaped node inheriting from nothing in this package -- not
    ``v2.Node``, not ``v2.Table``, not even ``object`` explicitly. Proves the
    protocols are the real contract, not the concrete classes: this satisfies
    ``protocols.Table`` by shape alone."""
    type = 'acme:scratch-table'
    entity = 'docs'

    def __init__(self, name, rows):
        self.name = name
        self.metadata = {}
        self.qualified_name = None
        self.provider = None
        self.defined_in = None
        self._rows = list(rows)

    def structural_edges(self):
        return []

    def count(self):
        return len(self._rows)

    @property
    def record_type(self):
        return type(self._rows[0]) if self._rows else None

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return self._rows[key]

    def lookup(self, ids):
        return {r[0]: r for r in self._rows if r[0] in ids}


class TestV2Protocols(unittest.TestCase):
    """Node/Resource/Table/Benchmark/Suite are structural contracts (protocols.py),
    not base classes -- inheriting from this package's concrete classes is a
    convenience, never a requirement to join the graph."""

    def test_concrete_classes_satisfy_their_protocols(self):
        self.assertIsInstance(graph['irds:antique-docs.txt'], v2.DataProtocol)
        self.assertIsInstance(graph['irds:antique-docs'], v2.TableProtocol)
        self.assertIsInstance(graph['irds:antique-test'], v2.BenchmarkProtocol)
        for node in (graph['irds:antique-docs.txt'], graph['irds:antique-docs'],
                    graph['irds:antique-test']):
            self.assertIsInstance(node, v2.NodeProtocol)

    def test_a_node_needs_no_particular_base_class(self):
        """The from-scratch table registers, edges and traverses
        exactly like one of ours -- because the registry only ever asks for
        the shape, never an isinstance check against a concrete class."""
        table = _FromScratchTable('scratch-table', [('r1', 'hello'), ('r2', 'world')])
        self.assertIsInstance(table, v2.TableProtocol)
        provider = v2.ManifestProvider('acme')
        provider.declare_graph_class('scratch-table')   # qualifies to 'acme:scratch-table'
        provider.register(table)
        self.assertEqual('acme:scratch-table', table.qualified_name)
        self.assertIn('acme:scratch-table', provider.nodes)
        g = v2.Graph([provider])
        self.assertEqual([], g.check())
        self.assertEqual([], g.closure('acme:scratch-table'))  # no structural edges

    def test_conformance_check_gives_a_clear_error(self):
        class NotANode:
            pass
        with self.assertRaisesRegex(TypeError, 'does not look like a Node: missing metadata, name'):
            v2.ManifestProvider('acme3').register(NotANode())


class TestV2Edges(unittest.TestCase):
    """The graph: edges, traversal, validity. No downloads."""

    def test_tables_declare_their_source_files(self):
        self.assertEqual(['irds:antique-docs.txt'],
                         graph.edges_of('irds:antique-docs')['irds:derivedFrom'])

    def test_benchmarks_declare_facet_edges(self):
        self.assertEqual(
            {'irds:docs': ['irds:antique-docs'], 'irds:queries': ['irds:antique-test-queries'],
             'irds:qrels': ['irds:antique-test-qrels']},
            graph.edges_of('irds:antique-test', structural_only=True))

    def test_derived_declares_parent_and_filter_source(self):
        """Benchmarks don't carry derived_from themselves -- the derived
        *tables* do, each naming exactly what it needs (including its
        filter's id source, folded into derived_from -- there's no separate
        filtered_by kind)."""
        edges = graph.edges_of('irds:antique-test-non-offensive')
        self.assertNotIn('irds:derivedFrom', edges)
        facet_edges = graph.edges_of('irds:antique-test-non-offensive-queries')
        self.assertCountEqual(['irds:antique-test-queries', 'irds:antique-test-disallow-list.txt'],
                              facet_edges['irds:derivedFrom'])

    def test_files_are_leaves(self):
        self.assertEqual({}, graph.edges_of('irds:antique-docs.txt'))

    def test_structural_vs_informational(self):
        self.assertTrue(v2.Edge('irds:docs', 'x').structural)
        self.assertFalse(v2.Edge('irds:same_corpus_as', 'x').structural)
        # this package declares no vocabulary except irds' own (other tests
        # may register their own kinds in the process-wide vocabulary index,
        # so check subset rather than exact equality)
        self.assertLessEqual(set(v2.STRUCTURAL_EDGES), v2.structural_kinds())

    def test_edge_kinds_must_be_declared(self):
        acme = v2.ManifestProvider('acme')
        with self.assertRaisesRegex(ValueError, 'undeclared edge kind'):
            acme.add_edge('irds:antique-docs', 'indexed_frm', 'acme:idx')
        with self.assertRaises(ValueError):      # disagreeing re-declaration
            irds.declare_graph_property('docs', structural=False)
        irds.declare_graph_property('docs', structural=True)   # agreeing one is a no-op

    def test_reverse_lookup(self):
        self.assertEqual(
            ['irds:antique-split200-train', 'irds:antique-split200-valid', 'irds:antique-test',
             'irds:antique-test-non-offensive', 'irds:antique-train'],
            graph.dependents('irds:antique-docs'))

    def test_closure(self):
        closure = graph.closure('irds:antique-test')
        for name in ('irds:antique-docs', 'irds:antique-docs.txt', 'irds:antique-test-qrels.qrel'):
            self.assertIn(name, closure)

    def test_graph_is_valid(self):
        self.assertEqual([], graph.check())

    def test_edges_are_an_adjacency_list(self):
        rows = irds.edge_rows()
        self.assertTrue(rows)
        self.assertEqual(3, len(rows[0]))


class TestV2Authority(unittest.TestCase):
    """Structural edges belong to the owner; informational edges to anyone."""

    def setUp(self):
        self.acme = v2.ManifestProvider('acme')
        self.acme.declare_graph_property('see_also', structural=False)
        self.graph = v2.Graph([irds, self.acme])

    def test_other_provider_may_add_informational_edge(self):
        self.acme.add_edge('irds:antique-docs', 'acme:see_also', 'acme:lite')
        self.assertEqual(['acme:lite'], self.graph.edges_of('irds:antique-docs')['acme:see_also'])
        self.assertEqual(['acme'], self.graph.edge_provenance('irds:antique-docs', 'acme:see_also', 'acme:lite'))
        # it is acme's contribution, not irds'
        self.assertIn(['irds:antique-docs', 'acme:see_also', 'acme:lite'], self.acme.edge_rows())
        self.assertNotIn('acme:see_also', graph.edges_of('irds:antique-docs'))

    def test_other_provider_may_not_add_structural_edge(self):
        with self.assertRaises(PermissionError):
            self.acme.add_edge('irds:antique-docs', 'irds:derivedFrom', 'acme:file')

    def test_loading_discovers_no_other_provider(self):
        """Resolving a node (``load`` asks for its structural closure) must not discover the catalog
        of any provider but the owner's: a live one would crawl or download."""
        asked = []

        class Live(v2.ManifestProvider):
            def discover_edges(self):
                asked.append(self.prefix)
                return super().discover_edges()
        g = v2.Graph([irds, Live('live')])
        g.load('irds:antique-test')
        g.dependencies('irds:antique-test')
        g.closure('irds:antique-test')
        self.assertEqual([], asked)
        g.edges_of('irds:antique-test')        # a full query does ask every provider
        self.assertEqual(['live'], asked)

    def test_third_party_edges_are_validated(self):
        self.acme.add_edge('irds:antique-docs', 'acme:see_also', 'acme:does-not-exist')
        self.assertIn('dangling: irds:antique-docs --acme:see_also--> acme:does-not-exist',
                      self.graph.check())

    def test_prefix_collision_is_an_error(self):
        with self.assertRaises(ValueError):
            v2.Graph([irds, v2.ManifestProvider('irds')])


class TestV2ResourceHashes(unittest.TestCase):
    """Resources declare one or more 'algo:hexdigest' hashes (the OCI/pip
    convention) -- md5= is a shorthand for the common single-hash case, and
    stays privileged for cache addressing/the v1 download check, but any
    number of additional algorithms can be declared and are all checked on
    first access."""

    def test_bare_hex_defaults_to_md5(self):
        r = v2.Resource('test-hash-bare', sources=['https://x/y'],
                        hashes=['0' * 32])
        self.assertEqual({'md5': '0' * 32}, r.hashes)
        self.assertEqual('0' * 32, r.md5)

    def test_multiple_algorithms(self):
        r = v2.Resource('test-hash-multi', sources=['https://x/y'],
                        hashes=['md5:' + 'a' * 32, 'sha256:' + 'b' * 64])
        self.assertEqual({'md5': 'a' * 32, 'sha256': 'b' * 64}, r.hashes)
        self.assertEqual({'type': 'file_hash'}, r.discovery_literals()['validation'])
        self.assertEqual(['md5:' + 'a' * 32, 'sha256:' + 'b' * 64], r.discovery_literals()['fileHash'])

    def test_dict_form(self):
        r = v2.Resource('test-hash-dict', sources=['https://x/y'],
                        hashes={'md5': 'a' * 32, 'sha1': 'c' * 40})
        self.assertEqual({'md5': 'a' * 32, 'sha1': 'c' * 40}, r.hashes)

    def test_md5_kwarg_is_shorthand(self):
        r = v2.Resource('test-hash-md5kw', sources=['https://x/y'], md5='f' * 32)
        self.assertEqual({'md5': 'f' * 32}, r.hashes)
        self.assertEqual({'type': 'file_hash'}, r.discovery_literals()['validation'])
        self.assertEqual(['md5:' + 'f' * 32], r.discovery_literals()['fileHash'])

    def test_explicit_hashes_win_over_md5_kwarg(self):
        r = v2.Resource('test-hash-precedence', sources=['https://x/y'],
                        md5='1' * 32, hashes=['md5:' + '2' * 32])
        self.assertEqual('2' * 32, r.md5)

    def test_unknown_algorithm_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, 'unknown hash algorithm'):
            v2.Resource('test-hash-bad-algo', sources=['https://x/y'],
                       hashes=['not-a-real-algo:abc'])

    def test_no_hash_is_legal(self):
        r = v2.Resource('test-hash-none', sources=['https://x/y'])
        self.assertEqual({}, r.hashes)
        self.assertIsNone(r.md5)


class TestV2ResourceShapes(unittest.TestCase):
    """`File`/`Directory`/`GitRepo` are Resource subclasses distinguishing
    what shape the bytes take -- a single blob, a tree addressed member-by-
    member, or a tree pinned to a git commit."""

    def test_file_behaves_exactly_like_a_bare_resource(self):
        """File adds no behavior of its own -- it's the same file-shaped
        Resource, just named for what it is now that Directory/GitRepo
        siblings exist."""
        f = v2.File('test-shapes-file', sources=['https://x/y'], md5='a' * 32)
        self.assertIsInstance(f, v2.Resource)
        self.assertEqual('a' * 32, f.md5)

    def test_directory_refuses_to_stream(self):
        d = v2.Directory('test-shapes-dir', sources=['https://x/y'])
        self.assertIsInstance(d, v2.Resource)
        with self.assertRaisesRegex(TypeError, 'not a single file'):
            d.stream()

    def test_git_repo_identity(self):
        pinned = v2.GitRepo('acme/repo', commit='deadbeef', pinned=True)
        self.assertEqual('acme/repo@deadbeef.git', pinned.name)
        self.assertEqual('acme/repo', pinned.metadata['repo'])

        unpinned = v2.GitRepo('acme/repo', commit='deadbeef', pinned=False)
        self.assertEqual('acme/repo.git', unpinned.name)

    def test_git_repo_file_addresses_one_file(self):
        repo = v2.GitRepo('acme/repo', commit='deadbeef', pinned=True)
        member = repo.file('data/file.jsonl')
        self.assertEqual("GitRepo('acme/repo@deadbeef.git').file('data/file.jsonl')",
                         repr(member))

    def test_directory_files_lists_the_tree_sorted_by_path(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            for rel in ('b.txt', 'a/z.txt', 'a/y/deep.txt', 'a.txt'):
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                open(os.path.join(root, rel), 'w').close()

            class Local(v2.Directory):
                def path(self, force=True):
                    return root
            self.assertEqual(['a.txt', 'a/y/deep.txt', 'a/z.txt', 'b.txt'],
                             list(Local('test-dir-files', sources=['https://x/y']).files()))


class TestV2IdsFromLines(unittest.TestCase):
    """`ids_from_lines`'s dependency edge must resolve through a pipeline
    (e.g. `.member()`/`.file()`, as used for a file extracted from a
    shared archive) down to the real Resource it bottoms out in -- a bare
    pipeline step has no name/identity of its own to hang a `derived_from`
    edge off of (found while porting nfcorpus, whose id-list files are
    members of the dataset's one shared tar.gz, not standalone downloads)."""

    def test_depends_on_a_bare_resource(self):
        r = v2.Resource('test-idsfromlines-bare', sources=['https://x/y'])
        ids_fn = v2.ids_from_lines(r)
        self.assertEqual((r,), ids_fn.depends_on)

    def test_depends_on_resolves_through_a_pipeline(self):
        r = v2.Resource('test-idsfromlines-archive.tar.gz', sources=['https://x/y'])
        pipeline = r.file('ids.txt')
        ids_fn = v2.ids_from_lines(pipeline)
        self.assertEqual((r,), ids_fn.depends_on)


class TestV2CachePath(unittest.TestCase):
    """A Resource's cache path (and a DocTable's docstore path) is
    `<home>/<provider>/<qualified name>` (not md5-keyed), computed lazily
    since it needs the registered provider/qualified_name -- and an existing
    v1 cache file/docstore is migrated to it, once, the first time it's
    found (see `sources.migrate_legacy`)."""

    def test_unregistered_resource_falls_back_to_local_bucket(self):
        r = v2.Resource('scratch-cache-path', sources=['https://x/y'])
        home = Path(ir_datasets.util.home_path())
        self.assertEqual(home / '_local' / 'scratch-cache-path', r.cache_path)

    def test_registered_resource_is_partitioned_by_provider(self):
        provider = v2.ManifestProvider('acmecache')
        r = v2.Resource('scratch-cache-path2', sources=['https://x/y'])
        provider.register(r)
        home = Path(ir_datasets.util.home_path())
        self.assertEqual(home / 'acmecache' / 'scratch-cache-path2', r.cache_path)

    def test_slash_in_name_nests_the_path(self):
        provider = v2.ManifestProvider('acmecache2')
        r = v2.Resource('owner/name/key', sources=['https://x/y'])
        provider.register(r)
        home = Path(ir_datasets.util.home_path())
        self.assertEqual(home / 'acmecache2' / 'owner' / 'name' / 'key', r.cache_path)

    def test_existing_v1_cache_is_migrated_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'IR_DATASETS_HOME': tmp}):
                home = Path(tmp)
                legacy_file = home / 'v1-dataset' / 'collection.tsv'
                legacy_file.parent.mkdir(parents=True)
                legacy_file.write_text('hello')
                md5 = 'a' * 32
                with mock.patch.object(
                        sourcesmod, '_legacy_cache',
                        return_value={md5: 'v1-dataset/collection.tsv'}):
                    provider = v2.ManifestProvider('acmecache3')
                    r = v2.Resource('migrated-file', sources=['https://x/y'], md5=md5)
                    provider.register(r)

                    existing = r.existing_path()

                    self.assertEqual(r.cache_path, existing)
                    self.assertTrue(r.cache_path.exists())
                    self.assertEqual('hello', r.cache_path.read_text())
                    # a compat symlink is left at the old v1 path
                    self.assertTrue(legacy_file.is_symlink())
                    self.assertEqual(r.cache_path.resolve(), legacy_file.resolve())
                    # already migrated -- the next call takes the fast path,
                    # no re-migration (and no error from a missing source)
                    self.assertEqual(existing, r.existing_path())

    def test_no_v1_cache_found_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'IR_DATASETS_HOME': tmp}):
                with mock.patch.object(sourcesmod, '_legacy_cache', return_value={}):
                    provider = v2.ManifestProvider('acmecache4')
                    r = v2.Resource('never-cached', sources=['https://x/y'], md5='b' * 32)
                    provider.register(r)
                    self.assertIsNone(r.existing_path())

    def test_v1_docstore_is_migrated_in_place(self):
        """A v1 docstore is a directory (PickleLz4FullStore), not a single
        file -- the same migrate_legacy move+symlink handles both."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'IR_DATASETS_HOME': tmp}):
                home = Path(tmp)
                legacy_file = home / 'v1-docs-dataset' / 'collection.tsv'
                legacy_file.parent.mkdir(parents=True)
                legacy_file.write_text('raw bytes')
                legacy_store = home / 'v1-docs-dataset' / 'collection.tsv.pklz4'
                legacy_store.mkdir()
                (legacy_store / 'data.lz4').write_text('store contents')
                md5 = 'c' * 32
                with mock.patch.object(
                        sourcesmod, '_legacy_cache',
                        return_value={md5: 'v1-docs-dataset/collection.tsv'}):
                    provider = v2.ManifestProvider('acmedocstore')
                    resource = v2.Resource('raw-docs', sources=['https://x/y'], md5=md5)
                    table = v2.TsvDocs('parsed-docs', source=resource)
                    provider.register(table)

                    path = table.docstore_path

                    self.assertEqual('parsed-docs.docstore.fmt1.pklz4', path.name)
                    self.assertTrue(path.is_dir())
                    self.assertTrue((path / 'data.lz4').exists())
                    # a compat symlink (directory-flavored) at the old v1 path
                    self.assertTrue(legacy_store.is_symlink())
                    self.assertEqual(path.resolve(), legacy_store.resolve())
                    # already migrated -- the next access takes the fast path
                    self.assertEqual(path, table.docstore_path)


class TestV2Citation(unittest.TestCase):
    """Citations and metrics are plain metadata for now -- not yet edges to
    paper/measure nodes (that's future work for a broader knowledge graph)."""

    def test_citation_is_a_generic_node_field(self):
        """Any node can carry one -- it's on the generic Node, not Benchmark."""
        node = v2.Resource('test-citation-file', sources=['https://x/y'],
                       citation='some-key')
        self.assertEqual('some-key', node.metadata['citation'])

    def test_benchmark_citation_and_metrics_are_metadata(self):
        node = graph['irds:antique-test']
        self.assertEqual('dblp:conf/ecir/HashemiAZC20', node.metadata['citation'])
        self.assertEqual(['P(rel=3)@10', 'nDCG@10', 'MAP(rel=3)', 'RR(rel=3)'],
                         node.metrics)
        self.assertEqual(node.metrics, node.metadata['metric'])

    def test_citation_accessor(self):
        self.assertEqual('dblp:conf/ecir/HashemiAZC20', v2.citation('irds:antique-test'))
        self.assertIsNone(v2.citation('irds:antique-docs'))

    def test_citation_and_metrics_are_reported_at_discovery(self):
        from ir_datasets.v2.vocabulary import row_for
        row = row_for(graph['irds:antique-test'])
        self.assertEqual('dblp:conf/ecir/HashemiAZC20', row['citation'])
        self.assertEqual(['P(rel=3)@10', 'nDCG@10', 'MAP(rel=3)', 'RR(rel=3)'], row['metric'])
        # not edges: no citation/measure-shaped edge kinds exist
        self.assertNotIn('citation', v2.edge_kinds())
        self.assertFalse([k for k in irds.edge_kinds if 'measure' in k or 'citation' in k])


class TestV2Suite(unittest.TestCase):
    """A Suite is a named, structural set of Benchmarks (e.g. BEIR)."""

    def setUp(self):
        self.provider = v2.ManifestProvider('acme')
        src = v2.Resource('acme-src', sources=['https://example.org/x.tsv'])
        docs = v2.TsvDocs('acme-docs', source=src)
        self.b1 = v2.Benchmark('acme-task1', docs=docs,
                               queries=v2.TsvQueries('acme-task1-queries', source=src))
        self.b2 = v2.Benchmark('acme-task2', docs=docs,
                               queries=v2.TsvQueries('acme-task2-queries', source=src))
        self.suite = v2.Suite('acme-suite', benchmarks=[self.b1, self.b2])
        self.provider.register(self.suite)
        self.graph = v2.Graph([self.provider])

    def test_type_is_suite(self):
        # Suite is a class irds declares vocabulary for; a third party may
        # still register instances of it into their own provider namespace.
        self.assertEqual(v2.SUITE, self.suite.type)
        self.assertEqual('irds:Suite', self.suite.type)
        self.assertEqual('acme:acme-suite', self.suite.qualified_name)

    def test_structural_edges_to_each_benchmark(self):
        edges = self.graph.edges_of('acme:acme-suite')
        self.assertEqual(sorted(['acme:acme-task1', 'acme:acme-task2']),
                         sorted(edges['irds:member']))
        self.assertTrue(v2.Edge('irds:member', 'x').structural)

    def test_membership_pulls_benchmarks_in_transitively(self):
        self.assertIn('acme:acme-task1', self.provider.nodes)
        self.assertIn('acme:acme-task2', self.provider.nodes)
        self.assertIn('acme:acme-docs', self.provider.nodes)   # shared corpus too

    def test_suite_is_a_dependent_of_its_benchmarks(self):
        self.assertIn('acme:acme-suite', self.graph.dependents('acme:acme-task1'))

    def test_closure_reaches_through_benchmark_to_table_to_file(self):
        closure = self.graph.closure('acme:acme-suite')
        self.assertIn('acme:acme-task1', closure)
        self.assertIn('acme:acme-docs', closure)
        self.assertIn('acme:acme-src', closure)

    def test_a_benchmark_may_belong_to_more_than_one_suite(self):
        other = v2.Suite('acme-other-suite', benchmarks=[self.b1])
        self.provider.register(other)
        self.assertEqual(sorted(['acme:acme-suite', 'acme:acme-other-suite']),
                         sorted(self.graph.referrers('acme:acme-task1', kind='irds:member')))

    def test_dangling_membership_is_reported(self):
        bad = v2.ManifestProvider('acme2')
        bad.register(v2.Suite('bad-suite', benchmarks=['acme2:no-such-benchmark']))
        g = v2.Graph([bad])
        self.assertIn('dangling: acme2:bad-suite --irds:member--> acme2:no-such-benchmark',
                      g.check())

    def test_names_may_be_used_instead_of_objects(self):
        provider2 = v2.ManifestProvider('acme3')
        src = v2.Resource('acme3-src', sources=['https://example.org/x.tsv'])
        b = v2.Benchmark('acme3-task', docs=v2.TsvDocs('acme3-docs', source=src))
        provider2.register(b)
        suite = v2.Suite('acme3-suite', benchmarks=['acme3-task'])  # bare name -> own namespace
        provider2.register(suite)
        g = v2.Graph([provider2])
        self.assertEqual(['acme3:acme3-task'], g.edges_of('acme3:acme3-suite')['irds:member'])


class TestV2Bundle(unittest.TestCase):
    """A Bundle: any nodes that belong together, with a hand-written summary."""

    def test_parts_are_structural_has_part_edges_of_any_kind(self):
        bundle = v2.Bundle('test-bundle-edges', parts=['irds:antique-docs', 'irds:antique-test'],
                           summary='# Antique\n\nQuestions.')
        self.assertEqual('irds:Bundle', bundle.type)
        self.assertEqual([('irds:bundleItem', 'irds:antique-docs'), ('irds:bundleItem', 'irds:antique-test')],
                         [(e.kind, e.target) for e in bundle.structural_edges()])
        self.assertFalse(v2.graph_properties()['irds:bundleItem']['structural'])   # a grouping, not a dependency
        self.assertEqual('# Antique\n\nQuestions.', bundle.metadata['summary'])

    def test_a_bundle_needs_nothing_its_parts_need_and_scales_to_many_parts(self):
        needy = v2.Resource('test-bundle-needy', sources=['https://x/y'], requires=['somepackage'])
        acme = v2.ManifestProvider('acme-bundle-big')
        parts = [f'acme-bundle-big:part-{i}' for i in range(50_000)]
        bundle = v2.Bundle('big', parts=['acme-bundle-big:needy', *parts])
        acme.register(needy.__class__('needy', sources=['https://x/y'], requires=['somepackage']), bundle)
        graph = v2.Graph()
        graph.add(acme)
        self.assertEqual((), graph.requires('acme-bundle-big:big'))           # parts' needs are not the bundle's
        self.assertEqual('somepackage', graph['acme-bundle-big:needy'].requires[0])
        import time
        started = time.time()
        self.assertEqual(50_001, len(graph.edges_of('acme-bundle-big:big')['irds:bundleItem']))
        self.assertLess(time.time() - started, 5)                              # linear, not quadratic

    def test_the_summary_is_a_declared_property_and_a_triple(self):
        from ir_datasets.v2.registry import row_triples
        bundle = v2.Bundle('test-bundle-triples', summary='text')
        irds_row = {'name': 'irds:test-bundle-triples', **bundle.metadata, 'type': bundle.type}
        triples = list(row_triples('irds:test-bundle-triples', irds_row))
        self.assertIn(('irds:test-bundle-triples', 'irds:summary'),
                      [(s, p) for s, p, _ in triples])

    def test_a_bundle_registers_and_loads_and_suite_is_unchanged(self):
        part = v2.Bundle('test-bundle-part')
        outer = v2.Bundle('test-bundle-outer', parts=[part])
        acme = v2.ManifestProvider('acme-bundle')
        acme.register(outer)
        self.assertEqual('acme-bundle:test-bundle-part', part.qualified_name)   # a part object is registered too
        self.assertEqual(['acme-bundle:test-bundle-part'],
                         acme.indexes()[0]['acme-bundle:test-bundle-outer']['irds:bundleItem'])
        self.assertEqual('irds:Suite', v2.Suite('test-suite-still').type)
        self.assertNotIsInstance(v2.Suite('test-suite-still-2'), v2.Bundle)


class TestV2Overview(unittest.TestCase):
    def test_an_overview_is_a_bundle_whose_prose_lives_in_its_documentation_page(self):
        overview = v2.Overview('test-overview', parts=['irds:antique-test'])
        self.assertEqual('irds:Overview', overview.type)
        self.assertIsInstance(overview, v2.Bundle)
        self.assertTrue(v2.is_subclass(overview.type, v2.BUNDLE))
        self.assertEqual([('irds:bundleItem', 'irds:antique-test')],
                         [(e.kind, e.target) for e in overview.structural_edges()])
        self.assertEqual('irds:Bundle', v2.Bundle('test-bare-bundle').type)


class TestV2LocalCopyHint(unittest.TestCase):
    def test_hint_for_large_file_names_the_cache_path(self):
        node = graph['irds:antique-docs.txt']          # 93.6 MB
        hint = local_copy_hint(node)
        self.assertIn(str(node.cache_path), hint)
        self.assertIn(node.sources[0].url, hint)

    def test_no_hint_for_small_file(self):
        self.assertIsNone(local_copy_hint(graph['irds:antique-test-queries.txt']))

    def test_no_hint_without_md5(self):
        node = v2.Resource('test-hint-no-md5', sources=['https://example.org/x'], size=10 ** 9)
        self.assertIsNone(local_copy_hint(node))


class TestV2Api(unittest.TestCase):
    """The default (property) API and the legacy (method) API must agree."""

    def setUp(self):
        self.benchmark = graph['irds:antique-test']

    def test_default_api(self):
        c = self.benchmark
        self.assertEqual(200, len(c.queries))
        self.assertEqual(6589, len(c.qrels))
        self.assertEqual('3990512', c.queries[0].query_id)
        self.assertEqual('1971899', c.queries[-1].query_id)
        self.assertEqual(['3990512', '714612'], [q.query_id for q in c.queries[:2]])
        self.assertEqual('GenericQuery', c.queries.record_type.__name__)
        self.assertEqual('en', c.docs.lang)
        self.assertEqual([1, 2, 3, 4], sorted(c.qrels.defs))

    def test_lookup(self):
        c = self.benchmark
        self.assertEqual('2020338_0', c.docs.lookup('2020338_0').doc_id)
        self.assertEqual({'2020338_0', '1424320_9'},
                         set(c.docs.lookup(['2020338_0', '1424320_9'])))
        self.assertEqual('3990512', c.queries.lookup('3990512').query_id)

    def test_legacy_api_agrees(self):
        c = self.benchmark
        self.assertEqual(len(c.queries), sum(1 for _ in c.queries_iter()))
        self.assertEqual(len(c.docs), c.docs_count())
        self.assertIs(c.docs.record_type, c.docs_cls())
        self.assertTrue(c.has_docs() and c.has_qrels())
        self.assertFalse(c.has_scoreddocs())

    def test_derived_blocks_support_both(self):
        d = graph['irds:antique-test-non-offensive']
        self.assertEqual(176, len(d.queries))
        self.assertEqual(176, sum(1 for _ in d.queries_iter()))
        self.assertEqual('3990512', d.queries[0].query_id)

    def test_tables_are_loadable_standalone(self):
        q = graph['irds:antique-test-queries']
        self.assertEqual(200, len(q))
        self.assertEqual('3990512', q[0].query_id)


if __name__ == '__main__':
    unittest.main()


class TestSourceRecords(unittest.TestCase):
    """A Resource's ``sources`` metadata is one JSON literal per source."""

    def _srcs(self, node):
        import json
        return [json.loads(s) for s in node.discovery_literals()['sources']]

    def test_url_mirror_and_manual_records(self):
        import tempfile
        r = v2.Resource('test-src-records', sources=[
            'https://example.org/x', v2.Source.mirror(),
            v2.Source.external('ds/file', old_locations=['old/file'], instructions='get it at {path}')], md5='a' * 32)
        url, mirror, manual = self._srcs(r)
        self.assertEqual({'type': 'url', 'order': 0, 'url': 'https://example.org/x'}, url)
        # the mirror is just another URL; a reader tells it apart by host name
        self.assertEqual(('url', 1, 'https://mirror.ir-datasets.com/' + 'a' * 32),
                         (mirror['type'], mirror['order'], mirror['url']))
        self.assertEqual(('manual', 2, 'ds/file', ['old/file']),
                         (manual['type'], manual['order'], manual['path'], manual['old_locations']))

    def test_local_resolves_under_external_with_legacy_fallback(self):
        import os, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as home:
            old = {k: os.environ.get(k) for k in ('IR_DATASETS_HOME', 'IR_DATASETS_EXTERNAL')}
            os.environ['IR_DATASETS_HOME'] = home
            os.environ.pop('IR_DATASETS_EXTERNAL', None)
            try:
                src = v2.Source.external('ds/file', old_locations=['old/file'])
                self.assertEqual(Path(home) / 'external' / 'ds' / 'file', src.local_path)
                (Path(home) / 'old').mkdir()
                (Path(home) / 'old' / 'file').write_text('x')
                self.assertEqual(Path(home) / 'old' / 'file', src.local_path)  # old location
                (Path(home) / 'external' / 'ds').mkdir(parents=True)
                (Path(home) / 'external' / 'ds' / 'file').write_text('y')
                self.assertEqual(Path(home) / 'external' / 'ds' / 'file', src.local_path)
            finally:
                for k, v in old.items():
                    if v is None: os.environ.pop(k, None)
                    else: os.environ[k] = v

    def test_external_status_lists_manual_resources(self):
        rows = v2.external_status()
        self.assertTrue(rows)
        self.assertTrue(all(len(r) == 4 for r in rows))
