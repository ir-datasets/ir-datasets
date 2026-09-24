"""Generic conformance harness for v2 nodes.

In v1 every family needed its own integration test with hand-pasted counts and
sample records (test/integration/antique.py and ~60 siblings). Here there is
ONE test: it walks the frozen manifests and checks every attested node. Adding a
dataset adds no test code -- only a `freeze --verify` run.

    python -m unittest test.v2_conformance               # all attested nodes
    python -m unittest test.v2_conformance.TestV2Graph    # structure only (no download)
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ir_datasets.util
import ir_datasets.v2 as v2
from ir_datasets.v2 import freeze, graph, irds
from ir_datasets.v2 import sources as sourcesmod
from ir_datasets.v2.sources import local_copy_hint
from ir_datasets.v2.verify import attested as _attested, verify


def _scratch(prefix='test'):
    """An isolated provider + graph; nothing here touches the real ones."""
    src = v2.Resource('scratch-file', sources=['https://example.org/x.tsv'])
    table = v2.TsvQueries('scratch-queries', source=src)
    provider = v2.Provider(prefix)
    return provider, src, table


class TestV2Graph(unittest.TestCase):
    """Structural checks against the frozen manifest: no data is downloaded."""

    def test_manifest_present(self):
        self.assertTrue(_attested(), 'no attested nodes; run freeze --verify')

    def test_every_node_resolves(self):
        for name in irds.manifest()['nodes']:
            with self.subTest(node=name):
                self.assertIsNotNone(graph[name])

    def test_node_types_are_known(self):
        known = {v2.RESOURCE, v2.TABLE, v2.BENCHMARK, v2.SUITE}
        for name in irds.manifest()['nodes']:
            with self.subTest(node=name):
                self.assertIn(graph.type_of(name), known)

    def test_exactly_four_node_types(self):
        """The paper's whole point: Resource, Table, Benchmark, Suite -- no more."""
        self.assertEqual({v2.RESOURCE, v2.TABLE, v2.BENCHMARK, v2.SUITE},
                         set(irds.types))

    def test_benchmark_edges_resolve(self):
        for name in irds.manifest()['nodes']:
            node = graph[name]
            if isinstance(node, v2.Benchmark):
                with self.subTest(node=name):
                    self.assertTrue(any(node.has(e) for e in v2.ENTITIES))

    def test_no_hidden_tier(self):
        """Every registered node is listed -- no leading-underscore convention
        (dropped; it was a listing rule with no model backing, not worth the
        two-tier mental model)."""
        antique_resource_names = [n for n in v2.list_datasets(type=v2.RESOURCE)
                                  if 'antique' in n]
        self.assertTrue(antique_resource_names)
        for name in antique_resource_names:
            self.assertIsNotNone(graph[name])

    def test_legacy_aliases_resolve(self):
        aliases = irds.manifest().get('aliases', {})
        self.assertTrue(aliases)
        for legacy, target in aliases.items():
            with self.subTest(alias=legacy):
                self.assertEqual(target, graph[legacy].qualified_name)

    def test_no_default_provider(self):
        """Every name has a home, and the home is in the name."""
        node = graph['irds:antique-test']
        self.assertEqual('irds:antique-test', node.qualified_name)
        with self.assertRaisesRegex(KeyError, 'not a qualified name'):
            graph['antique-test']
        self.assertIs(node, graph['antique/test'])              # legacy alias: explicit
        self.assertTrue(all(':' in n for n in v2.list_datasets()))

    def test_vocabulary_is_owned_and_qualified(self):
        self.assertEqual('irds:Benchmark', graph['irds:antique-test'].type)
        self.assertEqual('irds:Table', v2.TsvDocs.type)
        self.assertEqual('irds', v2.node_types()['irds:Table']['owner'])
        self.assertEqual('irds', v2.edge_kinds()['irds:derived_from']['owner'])
        tables = v2.list_datasets(type='irds:Table')
        self.assertIn('irds:antique-docs', tables)
        self.assertIn('irds:antique-test-qrels', tables)
        self.assertTrue(all(v2.graph.type_of(n) == 'irds:Table' for n in tables))
        # the manifest records the provider's vocabulary
        m = irds.manifest()
        self.assertIn('irds:Table', m['types'])
        self.assertTrue(m['edge_kinds']['irds:derived_from']['structural'])
        self.assertEqual(['deprecated', 'dua', 'lang', 'namespace'], m['defaultable'])

    def test_undeclared_type_cannot_register(self):
        class Rogue(v2.Node):
            type = 'acme:never-declared'
        with self.assertRaisesRegex(ValueError, 'undeclared node type'):
            v2.Provider('acme').register(Rogue('x'))
        class Typeless(v2.Node):
            pass
        with self.assertRaises(ValueError):
            v2.Provider('acme').register(Typeless('x'))

    def test_provider_declares_only_in_its_own_namespace(self):
        acme = v2.Provider('acme')
        self.assertEqual('acme:thing', acme.node_type('thing'))
        self.assertEqual('acme:made_from', acme.edge_kind('made_from', structural=True))
        with self.assertRaises(ValueError):
            acme.node_type('irds:thing')
        with self.assertRaises(ValueError):
            acme.edge_kind('irds:derived_from', structural=True)

    def test_providers_subscribe_to_the_default_graph(self):
        self.assertEqual(['hf', 'irds'], sorted(graph.providers))
        self.assertIs(v2.default_graph().providers, graph.providers)

    def test_known_names_are_opt_in_via_discover(self):
        """A provider's known_names() (a live/expensive lookup a dynamic
        provider may register) never leaks into the default list(); only
        discover=True merges it in."""
        provider, _, _ = _scratch('scratchknown')
        provider.register_known(lambda: ['made-up-name'])
        scratch_graph = v2.Graph([provider])
        self.assertNotIn('scratchknown:made-up-name', scratch_graph.list())
        self.assertIn('scratchknown:made-up-name', scratch_graph.list(discover=True))

    def test_entry_points_name_the_prefix(self):
        """pyproject declares each provider under the prefix it owns."""
        import tomllib
        with open(Path(__file__).parent.parent / 'pyproject.toml', 'rb') as f:
            eps = tomllib.load(f)['project']['entry-points'][v2.ENTRY_POINT_GROUP]
        for name, spec in eps.items():
            with self.subTest(entry_point=name):
                self.assertEqual(name, freeze.load_provider(spec).prefix)


class TestV2Registration(unittest.TestCase):
    """Construction is inert; registration is explicit and transitive."""

    def test_construction_does_not_register(self):
        _, src, table = _scratch()
        self.assertIsNone(table.provider)
        self.assertIsNone(table.qualified_name)
        self.assertNotIn('irds:scratch-queries', irds.nodes)
        # ...but the object is fully usable
        self.assertEqual([v2.Edge('irds:derived_from', src)], table.structural_edges())

    def test_register_applies_prefix_and_pulls_in_dependencies(self):
        provider, src, table = _scratch('acme')
        provider.register(table)
        self.assertEqual('acme:scratch-queries', table.qualified_name)
        self.assertIs(provider, table.provider)
        self.assertEqual(__name__, table.defined_in)
        # the Resource came along, and the edge was qualified
        self.assertEqual('acme:scratch-file', src.qualified_name)
        self.assertEqual({'irds:derived_from': ['acme:scratch-file']},   # acme uses irds' vocabulary
                         v2.Graph([provider]).edges_of('acme:scratch-queries'))

    def test_cannot_register_with_two_providers(self):
        provider, _, table = _scratch()
        provider.register(table)
        with self.assertRaises(ValueError):
            v2.Provider('other').register(table)

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
        # order isn't semantically meaningful (it depends on frozen-manifest
        # vs. live-registration merge order), so compare as sets
        self.assertCountEqual(['irds:antique-test-queries', 'irds:antique-test-disallow-list.txt'],
                              graph.edges_of(facet.qualified_name)['irds:derived_from'])

    def test_bare_names_and_kinds_mean_own_namespace(self):
        """The one shorthand, usable only inside a provider's own modules."""
        provider, src, table = _scratch('acme')
        provider.edge_kind('like', structural=False)
        provider.register(table)
        provider.add_edge('scratch-queries', 'like', 'other')
        self.assertEqual([['acme:scratch-queries', 'acme:like', 'acme:other']],
                         [r for r in provider.edge_rows() if r[1] == 'acme:like'])
        provider.alias('old/id', 'scratch-queries')
        self.assertEqual('acme:scratch-queries', provider.aliases['old/id'])

    def test_names_may_not_contain_colons(self):
        with self.assertRaises(ValueError):
            v2.Resource('acme:file', sources=['https://x/y'])
        with self.assertRaises(ValueError):
            v2.Provider('')


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

    def attest(self, *, verify=False, **options):
        return {'count': len(self._rows)} if verify else None

    def verify(self, frozen, **options):
        return [] if frozen.get('count') == len(self._rows) else ['mismatch']

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
        self.assertIsInstance(graph['irds:antique-docs.txt'], v2.ResourceProtocol)
        self.assertIsInstance(graph['irds:antique-docs'], v2.TableProtocol)
        self.assertIsInstance(graph['irds:antique-test'], v2.BenchmarkProtocol)
        for node in (graph['irds:antique-docs.txt'], graph['irds:antique-docs'],
                    graph['irds:antique-test']):
            self.assertIsInstance(node, v2.NodeProtocol)

    def test_a_node_needs_no_particular_base_class(self):
        """The from-scratch table registers, edges, freezes and traverses
        exactly like one of ours -- because the registry only ever asks for
        the shape, never an isinstance check against a concrete class."""
        table = _FromScratchTable('scratch-table', [('r1', 'hello'), ('r2', 'world')])
        self.assertIsInstance(table, v2.TableProtocol)
        provider = v2.Provider('acme')
        provider.node_type('scratch-table')   # qualifies to 'acme:scratch-table'
        provider.register(table)
        self.assertEqual('acme:scratch-table', table.qualified_name)
        self.assertIn('acme:scratch-table', provider.nodes)
        g = v2.Graph([provider])
        self.assertEqual([], g.check())
        self.assertEqual([], g.closure('acme:scratch-table'))  # no structural edges
        from ir_datasets.v2 import freeze as freeze_module
        row, has_attestation = freeze_module._attested_row(table, g, True, {})
        self.assertTrue(has_attestation)
        self.assertEqual(2, row['count'])

    def test_conformance_check_gives_a_clear_error(self):
        class NotANode:
            pass
        with self.assertRaisesRegex(TypeError, 'does not look like a Node'):
            v2.Provider('acme3').register(NotANode())


class TestV2Edges(unittest.TestCase):
    """The graph: edges, traversal, validity. No downloads."""

    def test_tables_declare_their_source_files(self):
        self.assertEqual(['irds:antique-docs.txt'],
                         graph.edges_of('irds:antique-docs')['irds:derived_from'])

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
        self.assertNotIn('irds:derived_from', edges)
        facet_edges = graph.edges_of('irds:antique-test-non-offensive-queries')
        self.assertCountEqual(['irds:antique-test-queries', 'irds:antique-test-disallow-list.txt'],
                              facet_edges['irds:derived_from'])

    def test_files_are_leaves(self):
        self.assertEqual({}, graph.edges_of('irds:antique-docs.txt'))

    def test_structural_vs_informational(self):
        self.assertTrue(v2.Edge('irds:docs', 'x').structural)
        self.assertFalse(v2.Edge('irds:same_corpus_as', 'x').structural)
        for kind in v2.INFORMATIONAL_EDGES:
            self.assertNotIn(kind, v2.STRUCTURAL_EDGES)
        # this package declares no vocabulary except irds' own (other tests
        # may register their own kinds in the process-wide vocabulary index,
        # so check subset rather than exact equality)
        self.assertLessEqual(set(v2.STRUCTURAL_EDGES), v2.structural_kinds())
        self.assertLessEqual(set(v2.INFORMATIONAL_EDGES), v2.informational_kinds())

    def test_edge_kinds_must_be_declared(self):
        acme = v2.Provider('acme')
        with self.assertRaisesRegex(ValueError, 'undeclared edge kind'):
            acme.add_edge('irds:antique-docs', 'indexed_frm', 'acme:idx')
        with self.assertRaises(ValueError):      # disagreeing re-declaration
            irds.edge_kind('docs', structural=False)
        irds.edge_kind('docs', structural=True)   # agreeing one is a no-op

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

    def test_manifest_has_adjacency_list_not_per_row_edges(self):
        manifest = irds.manifest()
        self.assertTrue(manifest['edges'])
        self.assertEqual(3, len(manifest['edges'][0]))
        self.assertFalse([r for r in manifest['nodes'].values() if 'edges' in r])

    def test_traversal_needs_no_imports(self):
        fresh = v2.Graph([v2.Provider('irds', package=irds.package,
                                      manifest_path=irds.manifest_path)])
        self.assertIn('irds:antique-test', fresh.dependents('irds:antique-docs'))
        self.assertFalse(fresh.providers['irds'].nodes)   # nothing was imported


class TestV2Authority(unittest.TestCase):
    """Structural edges belong to the owner; informational edges to anyone."""

    def setUp(self):
        self.acme = v2.Provider('acme')
        self.acme.edge_kind('see_also', structural=False)
        self.graph = v2.Graph([irds, self.acme])

    def test_other_provider_may_add_informational_edge(self):
        self.acme.add_edge('irds:antique-docs', 'acme:see_also', 'acme:lite')
        self.assertEqual(['acme:lite'], self.graph.edges_of('irds:antique-docs')['acme:see_also'])
        self.assertEqual(['acme'], self.graph.edge_provenance('irds:antique-docs', 'acme:see_also', 'acme:lite'))
        # it ships in acme's manifest, not irds'
        self.assertIn(['irds:antique-docs', 'acme:see_also', 'acme:lite'], self.acme.edge_rows())
        self.assertNotIn('acme:see_also', graph.edges_of('irds:antique-docs'))

    def test_other_provider_may_not_add_structural_edge(self):
        with self.assertRaises(PermissionError):
            self.acme.add_edge('irds:antique-docs', 'irds:derived_from', 'acme:file')

    def test_third_party_edges_are_validated(self):
        self.acme.add_edge('irds:antique-docs', 'acme:see_also', 'acme:does-not-exist')
        self.assertIn('dangling: irds:antique-docs --acme:see_also--> acme:does-not-exist',
                      self.graph.check())

    def test_prefix_collision_is_an_error(self):
        with self.assertRaises(ValueError):
            v2.Graph([irds, v2.Provider('irds')])


class TestV2ResourceHashes(unittest.TestCase):
    """Resources declare one or more 'algo:hexdigest' hashes (the OCI/pip
    convention) -- md5= is a shorthand for the common single-hash case, and
    stays privileged for cache addressing/the v1 download check, but any
    number of additional algorithms can be declared and are all checked at
    verify time."""

    def test_bare_hex_defaults_to_md5(self):
        r = v2.Resource('test-hash-bare', sources=['https://x/y'],
                        hashes=['0' * 32])
        self.assertEqual({'md5': '0' * 32}, r.hashes)
        self.assertEqual('0' * 32, r.md5)

    def test_multiple_algorithms(self):
        r = v2.Resource('test-hash-multi', sources=['https://x/y'],
                        hashes=['md5:' + 'a' * 32, 'sha256:' + 'b' * 64])
        self.assertEqual({'md5': 'a' * 32, 'sha256': 'b' * 64}, r.hashes)
        self.assertEqual(['md5:' + 'a' * 32, 'sha256:' + 'b' * 64],
                         r.metadata['hashes'])

    def test_dict_form(self):
        r = v2.Resource('test-hash-dict', sources=['https://x/y'],
                        hashes={'md5': 'a' * 32, 'sha1': 'c' * 40})
        self.assertEqual({'md5': 'a' * 32, 'sha1': 'c' * 40}, r.hashes)

    def test_md5_kwarg_is_shorthand(self):
        r = v2.Resource('test-hash-md5kw', sources=['https://x/y'], md5='f' * 32)
        self.assertEqual({'md5': 'f' * 32}, r.hashes)
        self.assertEqual(['md5:' + 'f' * 32], r.metadata['hashes'])

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
        self.assertEqual([], r.verify({}))
        self.assertIsNone(r.attest(verify=True))

    def test_verify_against_the_real_cached_file(self):
        """The antique docs file declares both md5 and sha256 -- exercises the
        real multi-hash streamed verification path against real cached bytes,
        not a mock."""
        r = graph['irds:antique-docs.txt']
        self.assertEqual({'md5', 'sha256'}, set(r.hashes))
        self.assertEqual([], r.verify({}))
        attestation = r.attest(verify=True)
        self.assertEqual(sorted(f'{a}:{h}' for a, h in r.hashes.items()),
                         attestation['hashes_confirmed'])

    def test_verify_catches_a_real_mismatch(self):
        r = graph['irds:antique-docs.txt']
        saved = r.hashes['sha256']
        r.hashes['sha256'] = '0' * 64
        try:
            divergences = r.verify({})
            self.assertEqual(1, len(divergences))
            self.assertIn('hash[sha256]', repr(divergences[0]))
        finally:
            r.hashes['sha256'] = saved


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

    def test_git_repo_identity_and_attest(self):
        class FakeGitRepo(v2.GitRepo):
            def _resolve_live_commit(self):
                return 'newsha'

        pinned = FakeGitRepo('acme/repo', commit='deadbeef', pinned=True)
        self.assertEqual('acme/repo@deadbeef.git', pinned.name)
        self.assertEqual({'commit': 'deadbeef'}, pinned.attest())
        self.assertEqual('acme/repo', pinned.metadata['repo'])
        # pinned: a commit SHA can't drift, so verify is trivially clean
        self.assertEqual([], pinned.verify({'commit': 'deadbeef'}))

        unpinned = FakeGitRepo('acme/repo', commit='deadbeef', pinned=False)
        self.assertEqual('acme/repo.git', unpinned.name)
        divergences = unpinned.verify({'commit': 'deadbeef'})
        self.assertEqual(1, len(divergences))
        self.assertIn('commit', repr(divergences[0]))

    def test_git_repo_verify_without_live_commit_check_is_a_noop(self):
        """A subclass that doesn't override `_resolve_live_commit` can't
        detect drift -- verify() reports nothing rather than raising."""
        repo = v2.GitRepo('acme/repo', commit='deadbeef', pinned=False)
        self.assertEqual([], repo.verify({'commit': 'someothersha'}))

    def test_git_repo_relative_addresses_a_member(self):
        repo = v2.GitRepo('acme/repo', commit='deadbeef', pinned=True)
        member = repo.relative('data/file.jsonl')
        self.assertEqual("GitRepo('acme/repo@deadbeef.git').relative('data/file.jsonl')",
                         repr(member))


class TestV2IdsFromLines(unittest.TestCase):
    """`ids_from_lines`'s dependency edge must resolve through a pipeline
    (e.g. `.member()`/`.relative()`, as used for a file extracted from a
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
        pipeline = r.relative('ids.txt')
        ids_fn = v2.ids_from_lines(pipeline)
        self.assertEqual((r,), ids_fn.depends_on)


class TestV2CachePath(unittest.TestCase):
    """A Resource's cache path (and a Docs' docstore path) is
    `<home>/<provider>/<qualified name>` (not md5-keyed), computed lazily
    since it needs the registered provider/qualified_name -- and an existing
    v1 cache file/docstore is migrated to it, once, the first time it's
    found (see `sources.migrate_legacy`)."""

    def test_unregistered_resource_falls_back_to_local_bucket(self):
        r = v2.Resource('scratch-cache-path', sources=['https://x/y'])
        home = Path(ir_datasets.util.home_path())
        self.assertEqual(home / '_local' / 'scratch-cache-path', r.cache_path)

    def test_registered_resource_is_partitioned_by_provider(self):
        provider = v2.Provider('acmecache')
        r = v2.Resource('scratch-cache-path2', sources=['https://x/y'])
        provider.register(r)
        home = Path(ir_datasets.util.home_path())
        self.assertEqual(home / 'acmecache' / 'scratch-cache-path2', r.cache_path)

    def test_slash_in_name_nests_the_path(self):
        provider = v2.Provider('acmecache2')
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
                    provider = v2.Provider('acmecache3')
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
                    provider = v2.Provider('acmecache4')
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
                    provider = v2.Provider('acmedocstore')
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
        self.assertEqual(node.metrics, node.metadata['metrics'])

    def test_citation_accessor(self):
        self.assertEqual('dblp:conf/ecir/HashemiAZC20', v2.citation('irds:antique-test'))
        self.assertIsNone(v2.citation('irds:antique-docs'))

    def test_citation_survives_freeze(self):
        row = irds.manifest()['nodes']['irds:antique-test']
        self.assertEqual('dblp:conf/ecir/HashemiAZC20', row['citation'])
        self.assertEqual(['P(rel=3)@10', 'nDCG@10', 'MAP(rel=3)', 'RR(rel=3)'], row['metrics'])
        # not edges: no citation/measure-shaped edge kinds exist
        self.assertNotIn('citation', v2.edge_kinds())
        self.assertFalse([k for k in irds.edge_kinds if 'measure' in k or 'citation' in k])


class TestV2Suite(unittest.TestCase):
    """A Suite is a named, structural set of Benchmarks (e.g. BEIR)."""

    def setUp(self):
        self.provider = v2.Provider('acme')
        src = v2.Resource('acme-src', sources=['https://example.org/x.tsv'])
        docs = v2.TsvDocs('acme-docs', source=src)
        self.b1 = v2.Benchmark('acme-task1', docs=docs,
                               queries=v2.TsvQueries('acme-task1-queries', source=src))
        self.b2 = v2.Benchmark('acme-task2', docs=docs,
                               queries=v2.TsvQueries('acme-task2-queries', source=src))
        self.suite = v2.Suite('acme-suite', benchmarks=[self.b1, self.b2],
                              desc='A toy suite for testing.')
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
                         sorted(edges['irds:benchmark']))
        self.assertTrue(v2.Edge('irds:benchmark', 'x').structural)

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
                         sorted(self.graph.referrers('acme:acme-task1', kind='irds:benchmark')))

    def test_dangling_membership_is_reported(self):
        bad = v2.Provider('acme2')
        bad.register(v2.Suite('bad-suite', benchmarks=['acme2:no-such-benchmark']))
        g = v2.Graph([bad])
        self.assertIn('dangling: acme2:bad-suite --irds:benchmark--> acme2:no-such-benchmark',
                      g.check())

    def test_names_may_be_used_instead_of_objects(self):
        provider2 = v2.Provider('acme3')
        src = v2.Resource('acme3-src', sources=['https://example.org/x.tsv'])
        b = v2.Benchmark('acme3-task', docs=v2.TsvDocs('acme3-docs', source=src))
        provider2.register(b)
        suite = v2.Suite('acme3-suite', benchmarks=['acme3-task'])  # bare name -> own namespace
        provider2.register(suite)
        g = v2.Graph([provider2])
        self.assertEqual(['acme3:acme3-task'], g.edges_of('acme3:acme3-suite')['irds:benchmark'])


class TestV2Defaults(unittest.TestCase):
    """defaults() fills in shared descriptive fields, and nothing else."""

    def test_defaults_reached_every_node(self):
        for name in ['irds:antique-docs.txt', 'irds:antique-test-qrels.qrel']:
            self.assertTrue(graph[name].dua, name)
        for name in ['irds:antique-docs', 'irds:antique-test-queries', 'irds:antique-test-qrels']:
            self.assertEqual('en', graph[name].lang, name)

    def test_explicit_argument_wins(self):
        src = v2.Resource('test-defaults-src', sources=['https://example.org/x'])
        with irds.defaults(lang='en'):
            node = v2.TsvQueries('test-defaults-explicit', source=src, lang='de')
        self.assertEqual('de', node.lang)

    def test_identity_and_data_fields_are_rejected(self):
        self.assertEqual({'dua', 'lang', 'namespace', 'deprecated'}, irds.defaultable_fields)
        for field in ('md5', 'source', 'defs', 'name', 'parser'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                with irds.defaults(**{field: 'x'}):
                    pass

    def test_defaults_are_the_providers(self):
        """Another provider may not default fields it has not declared, even
        ones irds did."""
        acme = v2.Provider('acme')
        with self.assertRaisesRegex(ValueError, 'cannot default lang'):
            with acme.defaults(lang='en'):
                pass
        acme.defaultable('lang')
        src = v2.Resource('test-acme-src', sources=['https://example.org/x'])
        with acme.defaults(lang='fr'):
            self.assertEqual('fr', v2.TsvQueries('test-acme-q', source=src).lang)

    def test_defaults_do_not_leak_past_the_block(self):
        src = v2.Resource('test-defaults-leak-src', sources=['https://example.org/x'])
        with irds.defaults(lang='zh'):
            pass
        self.assertIsNone(v2.TsvQueries('test-defaults-leak', source=src).lang)


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


class TestV2Attestations(unittest.TestCase):
    """Data checks: materializes each attested node (downloads as needed)."""

    def test_attestations(self):
        for name in _attested():
            with self.subTest(node=name):
                self.assertEqual([], verify(name))


if __name__ == '__main__':
    unittest.main()
