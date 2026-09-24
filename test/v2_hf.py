"""Unit tests for the ``hf:`` dynamic provider (ir_datasets/v2/hf_provider.py).

No network: every test patches the module's own network-touching functions
(``_resolve_commit``, ``_card``, ``_hf_lib``) with fixtures, matching how the
rest of v2's tests keep structural checks free of downloads. Tests that don't
care about column resolution never trigger it (it's lazy -- see
``_HfFileParser``/``_HfDatasetsParser``); the ones that do also mock
``HfDataset.path``/``_rows_of`` so peeking a table's columns needs no real
file either. Each test uses its own fake repo id so the module-level caches
(``_card_cache``, ``_repo_cache``, ``_table_cache``) and the ``hf`` provider's
registry never collide across tests.
"""
import itertools
import tempfile
import unittest
from unittest import mock

import yaml

import ir_datasets.v2 as v2
from ir_datasets.v2 import hf_provider as hfm

_counter = itertools.count()


def _repo(name):
    """A fresh, never-before-used fake repo id for this test."""
    return f'test-owner/{name}-{next(_counter)}'


def _docs_table(entity_key='corpus', entity='docs'):
    # column_map: omitted -- these are purely structural tests (existence,
    # identity, naming) that never access .record_type or iterate, so column
    # resolution (which needs I/O) is never triggered.
    return {entity_key: {'entity': entity, 'file': f'data/{entity_key}.jsonl'}}


def _write_readme(text):
    fd, path = tempfile.mkstemp(suffix='.md')
    with open(fd, 'w', encoding='utf-8') as fout:
        fout.write(text)
    return path


class TestV2Hf(unittest.TestCase):
    def _patch(self, target, **kwargs):
        ctx = mock.patch.object(hfm, target, **kwargs)
        obj = ctx.start()
        self.addCleanup(ctx.stop)
        return obj

    def _mock(self, card_by_repo, commit='deadbeef'):
        """Fixture-backed replacements for every network call the resolver
        makes."""
        self._patch('_resolve_commit', side_effect=lambda repo, revision: commit)
        self._patch('_card', side_effect=lambda repo, commit: card_by_repo[repo])

    def _mock_rows(self, rows):
        """Makes ``HfDataset.path()``/``_rows_of(...)`` return canned rows,
        for tests that need column resolution (peeking a table's source)
        without any real file or network access. A fresh iterator each call
        -- ``_rows_of`` is invoked once to peek columns and again to actually
        iterate, and a single shared iterator would be exhausted by the peek."""
        ctx = mock.patch.object(hfm.HfDataset, 'path', return_value='/fake/path')
        ctx.start()
        self.addCleanup(ctx.stop)
        self._patch('_rows_of', side_effect=lambda path: iter(rows))

    def test_fragment_resolves_a_table(self):
        repo = _repo('single')
        self._mock({repo: {'tags': [], 'tables': _docs_table(), 'benchmarks': {},
                          'default': None}})
        node = v2.graph[f'hf:{repo}/corpus']
        self.assertIsInstance(node, v2.Docs)
        self.assertEqual(node.entity, 'docs')

    def test_no_default_is_a_key_error_even_with_one_table(self):
        """`default:` is never inferred -- not even when there's exactly one
        table and no ambiguity to speak of."""
        repo = _repo('nodefault')
        self._mock({repo: {'tags': [], 'tables': _docs_table(), 'benchmarks': {},
                          'default': None}})
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}']
        # the table is still reachable by fragment
        self.assertEqual(v2.graph[f'hf:{repo}/corpus'].entity, 'docs')

    def test_default_names_a_table_key(self):
        repo = _repo('defaulttable')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries')}
        self._mock({repo: {'tags': [], 'tables': tables, 'benchmarks': {},
                          'default': 'corpus'}})
        node = v2.graph[f'hf:{repo}']
        self.assertEqual(node.entity, 'docs')
        # a direct identity, not the bare name aliased to a fragmented one --
        # so the default key has no separate /fragment of its own
        self.assertEqual(node.qualified_name, f'hf:{repo}')
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}/corpus']
        # the OTHER table is unaffected, still reachable by its own fragment
        self.assertEqual(v2.graph[f'hf:{repo}/test'].entity, 'queries')

    def test_no_column_map_takes_every_column(self):
        """No column_map: -> the record type isn't known until the source is
        actually peeked at (needs I/O), so it reflects whatever columns are
        really there -- not assumed to be ir_datasets' minimal (doc_id, text)."""
        repo = _repo('nocolumnmap')
        self._mock({repo: {'tags': [],
                          'tables': {'corpus': {'entity': 'docs', 'file': 'data/x.jsonl'}},
                          'benchmarks': {}, 'default': None}})
        node = v2.graph[f'hf:{repo}/corpus']
        parser = node.parser
        # not yet resolved -- no I/O has happened
        self.assertIsNone(parser.columns)
        self._mock_rows([{'doc_id': 'd1', 'title': 'T', 'text': 'hello', 'extra': 'x'}])
        self.assertEqual(node.record_type._fields, ('doc_id', 'title', 'text', 'extra'))
        self.assertEqual(parser.columns, {'doc_id': 'doc_id', 'title': 'title',
                                          'text': 'text', 'extra': 'extra'})

    def test_column_map_renames_and_keeps_unmentioned_columns(self):
        """column_map: only needs to name the exceptions -- a rename or two
        -- not the whole schema; anything unmentioned passes through as-is."""
        repo = _repo('colmap')
        self._mock({repo: {'tags': [], 'default': 'corpus',
                          'tables': {'corpus': {'entity': 'docs', 'file': 'data/x.jsonl',
                                                'column_map': {'_id': 'doc_id',
                                                               'abstract': 'text'}}},
                          'benchmarks': {}}})
        self._mock_rows([{'_id': 'd1', 'title': 'T', 'abstract': 'hello'}])
        node = v2.graph[f'hf:{repo}']
        self.assertEqual(node.record_type._fields, ('doc_id', 'title', 'text'))
        rec = next(iter(node))
        self.assertEqual(rec, node.record_type(doc_id='d1', title='T', text='hello'))

    def test_column_map_null_drops_a_column(self):
        repo = _repo('coldrop')
        self._mock({repo: {'tags': [], 'default': 'corpus',
                          'tables': {'corpus': {'entity': 'docs', 'file': 'data/x.jsonl',
                                                'column_map': {'_id': 'doc_id', 'junk': None}}},
                          'benchmarks': {}}})
        self._mock_rows([{'_id': 'd1', 'junk': 'ignore-me', 'text': 'hello'}])
        node = v2.graph[f'hf:{repo}']
        self.assertEqual(node.record_type._fields, ('doc_id', 'text'))

    def test_column_map_unknown_column_is_an_error(self):
        repo = _repo('colunknown')
        self._mock({repo: {'tags': [], 'default': 'corpus',
                          'tables': {'corpus': {'entity': 'docs', 'file': 'data/x.jsonl',
                                                'column_map': {'nope': 'doc_id'}}},
                          'benchmarks': {}}})
        self._mock_rows([{'_id': 'd1', 'text': 'hello'}])
        node = v2.graph[f'hf:{repo}']
        with self.assertRaises(ValueError):
            node.record_type

    def test_missing_required_id_field_is_an_error(self):
        """Whatever the final field set is (declared or auto-discovered), the
        entity's required id field(s) must be in it."""
        repo = _repo('missingid')
        self._mock({repo: {'tags': [], 'default': 'corpus',
                          'tables': {'corpus': {'entity': 'docs', 'file': 'data/x.jsonl'}},
                          'benchmarks': {}}})
        self._mock_rows([{'text': 'hello'}])
        node = v2.graph[f'hf:{repo}']
        with self.assertRaises(ValueError):
            node.record_type

    def test_no_file_defaults_to_datasets_load_dataset(self):
        repo = _repo('viadatasets')
        self._mock({repo: {'tags': [],
                          'tables': {'train': {'entity': 'docs'}},
                          'benchmarks': {}, 'default': None}})
        node = v2.graph[f'hf:{repo}/train']
        self.assertIsInstance(node.parser, hfm._HfDatasetsParser)
        self.assertEqual(node.parser.config, None)
        self.assertEqual(node.parser.split, 'train')
        self.assertEqual(node.parser.repo, repo)

    def test_no_file_config_split_from_compound_key(self):
        repo = _repo('viadatasets-cfg')
        self._mock({repo: {'tags': [],
                          'tables': {'en/train': {'entity': 'docs'}},
                          'benchmarks': {}, 'default': None}})
        node = v2.graph[f'hf:{repo}/en/train']
        self.assertIsInstance(node.parser, hfm._HfDatasetsParser)
        self.assertEqual(node.parser.config, 'en')
        self.assertEqual(node.parser.split, 'train')

    def test_explicit_split_overrides_the_key(self):
        """The table key is just an addressable name -- it need not equal
        the Hub's own split name; `split:`/`config:` override it."""
        repo = _repo('splitoverride')
        self._mock({repo: {'tags': [],
                          'tables': {'docs': {'entity': 'docs', 'split': 'csl'}},
                          'benchmarks': {}, 'default': None}})
        node = v2.graph[f'hf:{repo}/docs']
        self.assertEqual(node.parser.split, 'csl')
        self.assertEqual(node.parser.config, None)

    def test_unreachable_tables_without_a_benchmark_or_default(self):
        repo = _repo('multi')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries')}
        self._mock({repo: {'tags': [], 'tables': tables, 'benchmarks': {},
                          'default': None}})
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}']
        self.assertEqual(v2.graph[f'hf:{repo}/corpus'].entity, 'docs')
        self.assertEqual(v2.graph[f'hf:{repo}/test'].entity, 'queries')

    def test_benchmark_facets_reachable_without_default(self):
        repo = _repo('bench')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries'),
                 **_docs_table('qrels', 'qrels')}
        card = {'tags': ['ir-datasets'], 'tables': tables, 'default': None,
               'benchmarks': {'eval': {'docs': 'corpus', 'queries': 'test', 'qrels': 'qrels',
                                       'metrics': ['nDCG@10']}}}
        self._mock({repo: card})
        # a benchmarks: block alone doesn't make the bare name resolve to it
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}']
        docs = v2.graph[f'hf:{repo}/docs']
        self.assertEqual(docs.entity, 'docs')
        # the raw table key is reachable too, and is the same node
        self.assertIs(v2.graph[f'hf:{repo}/corpus'], docs)
        # so is the benchmark itself, by its own benchmarks: key
        benchmark = v2.graph[f'hf:{repo}/eval']
        self.assertIsInstance(benchmark, v2.Benchmark)
        self.assertIs(benchmark.docs, docs)

    def test_default_benchmark_is_the_bare_name(self):
        repo = _repo('benchdefault')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries'),
                 **_docs_table('qrels', 'qrels')}
        card = {'tags': ['ir-datasets'], 'tables': tables, 'default': 'eval',
               'benchmarks': {'eval': {'docs': 'corpus', 'queries': 'test', 'qrels': 'qrels',
                                       'metrics': ['nDCG@10']}}}
        self._mock({repo: card})
        node = v2.graph[f'hf:{repo}']
        self.assertIsInstance(node, v2.Benchmark)
        self.assertEqual(node.metrics, ['nDCG@10'])
        self.assertEqual(node.qualified_name, f'hf:{repo}')
        self.assertIs(v2.graph[f'hf:{repo}/docs'], node.docs)
        self.assertIs(v2.graph[f'hf:{repo}/queries'], node.queries)
        self.assertIs(v2.graph[f'hf:{repo}/qrels'], node.qrels)
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}/eval']

    def test_default_names_a_benchmark_facet(self):
        """`default:` need not be the benchmark -- it can pick out just one
        facet as the repo's headline node. The facet NAME (docs) loses its
        own fragment, but the underlying raw key (corpus) -- a different
        string -- still works, resolving to the same, bare-named node."""
        repo = _repo('benchfacetdefault')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries')}
        card = {'tags': [], 'tables': tables, 'default': 'docs',
               'benchmarks': {'test': {'docs': 'corpus', 'queries': 'test'}}}
        self._mock({repo: card})
        node = v2.graph[f'hf:{repo}']
        self.assertIsInstance(node, v2.Docs)
        self.assertEqual(node.qualified_name, f'hf:{repo}')
        self.assertIs(node, v2.graph[f'hf:{repo}/corpus'])
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}/docs']

    def test_multiple_benchmarks_share_tables(self):
        """Two named benchmarks, each addressable on its own; a bare facet
        name is ambiguous once there's more than one."""
        repo = _repo('multibench')
        tables = {**_docs_table('corpus', 'docs'),
                 **_docs_table('train-q', 'queries'), **_docs_table('train-qr', 'qrels'),
                 **_docs_table('test-q', 'queries'), **_docs_table('test-qr', 'qrels')}
        card = {'tags': [], 'tables': tables, 'default': 'test',
               'benchmarks': {
                   'train': {'docs': 'corpus', 'queries': 'train-q', 'qrels': 'train-qr'},
                   'test': {'docs': 'corpus', 'queries': 'test-q', 'qrels': 'test-qr'},
               }}
        self._mock({repo: card})
        default_node = v2.graph[f'hf:{repo}']
        self.assertIsInstance(default_node, v2.Benchmark)
        self.assertEqual(default_node.qualified_name, f'hf:{repo}')
        train = v2.graph[f'hf:{repo}/train']
        self.assertIsInstance(train, v2.Benchmark)
        self.assertEqual(train.qualified_name, f'hf:{repo}/train')
        self.assertIsNot(train, default_node)
        self.assertIs(train.docs, default_node.docs)
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}/test']  # test is the default; no separate fragment
        # a bare facet fragment is ambiguous with two benchmarks declared
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}/docs']

    def test_benchmark_facet_can_reference_another_repo(self):
        """A facet need not be a local `tables:` key -- a fully qualified
        name (here, another `hf:` repo) is resolved lazily against the graph
        instead, exactly like `Benchmark`'s own name-string facets."""
        corpus_repo = _repo('corpus')
        eval_repo = _repo('borrower')
        corpus_card = {'tags': [], 'tables': _docs_table('corpus', 'docs'),
                       'benchmarks': {}, 'default': 'corpus'}
        tables = {**_docs_table('test-q', 'queries'), **_docs_table('test-qr', 'qrels')}
        eval_card = {'tags': [], 'tables': tables, 'default': None,
                    'benchmarks': {'eval': {
                        'docs': f'hf:{corpus_repo}', 'queries': 'test-q', 'qrels': 'test-qr'}}}
        self._mock({corpus_repo: corpus_card, eval_repo: eval_card})
        benchmark = v2.graph[f'hf:{eval_repo}/eval']
        self.assertIsInstance(benchmark, v2.Benchmark)
        self.assertEqual(benchmark.docs.qualified_name, f'hf:{corpus_repo}')
        self.assertIs(benchmark.docs, v2.graph[f'hf:{corpus_repo}'])

    def test_default_facet_shorthand_can_reference_another_repo(self):
        """The single-benchmark `/docs` shorthand (see
        `test_default_names_a_benchmark_facet`) also accepts a fully
        qualified facet value, not just a local raw key."""
        corpus_repo = _repo('corpus2')
        eval_repo = _repo('borrower2')
        corpus_card = {'tags': [], 'tables': _docs_table('corpus', 'docs'),
                       'benchmarks': {}, 'default': 'corpus'}
        eval_card = {'tags': [], 'tables': _docs_table('test-q', 'queries'),
                    'benchmarks': {'eval': {
                        'docs': f'hf:{corpus_repo}', 'queries': 'test-q'}}}
        self._mock({corpus_repo: corpus_card, eval_repo: eval_card})
        docs = v2.graph[f'hf:{eval_repo}/docs']
        self.assertEqual(docs.qualified_name, f'hf:{corpus_repo}')
        self.assertIs(docs, v2.graph[f'hf:{corpus_repo}'])
        # the fragment shorthand aliases to the same, externally-owned node
        self.assertIs(v2.graph[f'hf:{eval_repo}/docs'], docs)

    def test_both_card_keys_present_is_an_error(self):
        repo = _repo('dupe')
        readme = ('---\n' + yaml.safe_dump({
            'ir_datasets': {'tables': _docs_table()},
            'ir-datasets': {'tables': _docs_table()},
        }) + '---\nbody\n')
        self._patch('_hf_lib')
        hfm._hf_lib().hf_hub_download.return_value = _write_readme(readme)
        with self.assertRaises(ValueError):
            hfm._card(repo, 'deadbeef')

    def test_unannotated_repo_is_a_key_error(self):
        repo = _repo('bare')
        readme = '---\nlicense: mit\n---\nno ir_datasets block here\n'
        self._patch('_hf_lib')
        hfm._hf_lib().hf_hub_download.return_value = _write_readme(readme)
        with self.assertRaises(KeyError):
            hfm._card(repo, 'deadbeef')

    def test_revision_distinguishes_nodes(self):
        repo = _repo('rev')
        card = {'tags': [], 'tables': _docs_table(), 'benchmarks': {}}
        commits = {None: 'deadbeef', 'v1': 'abc123'}
        self._patch('_resolve_commit', side_effect=lambda r, revision: commits[revision])
        self._patch('_card', return_value=card)
        bare = v2.graph[f'hf:{repo}/corpus']
        pinned = v2.graph[f'hf:{repo}@v1/corpus']
        self.assertNotEqual(bare.qualified_name, pinned.qualified_name)
        self.assertIsNot(bare, pinned)

    def _resource_of(self, table):
        source = table.source
        return source._parent if hasattr(source, '_parent') else source

    def test_shared_hf_dataset_across_facets(self):
        repo = _repo('reuse')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries')}
        self._mock({repo: {'tags': [], 'tables': tables, 'benchmarks': {}}})
        docs = v2.graph[f'hf:{repo}/corpus']
        queries = v2.graph[f'hf:{repo}/test']
        self.assertIs(self._resource_of(docs), self._resource_of(queries))
        self.assertIsInstance(self._resource_of(docs), hfm.HfDataset)

    def test_graph_check_clean(self):
        repo = _repo('checkclean')
        tables = {**_docs_table('corpus', 'docs'), **_docs_table('test', 'queries'),
                 **_docs_table('qrels', 'qrels')}
        card = {'tags': [], 'tables': tables, 'default': 'eval',
               'benchmarks': {'eval': {'docs': 'corpus', 'queries': 'test', 'qrels': 'qrels'}}}
        self._mock({repo: card})
        node = v2.graph[f'hf:{repo}']
        self.assertIsInstance(node, v2.Benchmark)
        self.assertEqual(v2.graph.check(), [])

    def test_hf_dataset_name_disambiguates_from_table(self):
        repo = _repo('gitname')
        self._mock({repo: {'tags': [], 'tables': _docs_table(), 'benchmarks': {},
                          'default': 'corpus'}})
        table = v2.graph[f'hf:{repo}']
        hf_dataset = self._resource_of(table)
        # the default table's identity IS the bare name -- .git is what
        # keeps the HfDataset resource from colliding with it
        self.assertEqual(table.qualified_name, f'hf:{repo}')
        self.assertEqual(hf_dataset.qualified_name, f'hf:{repo}.git')
        self.assertNotEqual(table.qualified_name, hf_dataset.qualified_name)

    def test_search_lists_hub_repos_tagged_ir_datasets(self):
        """``search(None)`` -- what ``known()`` uses for its repo listing --
        is the full tag listing, no text filter, returning bare repo ids."""
        fake_hub = mock.Mock()
        fake_hub.HfApi.return_value.list_datasets.return_value = [
            mock.Mock(id='owner/a'), mock.Mock(id='owner/b')]
        self._patch('_hf_lib', return_value=fake_hub)
        self.assertEqual(hfm.search(None), ['owner/a', 'owner/b'])
        fake_hub.HfApi.return_value.list_datasets.assert_called_once_with(
            search=None, filter=hfm.TAG)

    def test_known_expands_each_tagged_repo_into_its_addressable_names(self):
        """``known()`` doesn't just list repo ids -- it resolves each repo's
        tables and benchmarks too (registering them into the graph), so a
        listing built from it already has real types instead of needing a
        separate visit per node to discover them."""
        repo = _repo('discoverable')
        card = {'tags': ['ir-datasets'], 'default': 'corpus',
               'tables': {**_docs_table('corpus', 'docs'), **_docs_table('extra', 'queries')},
               'benchmarks': {}}
        self._patch('search', return_value=[repo])
        self._mock({repo: card})
        names = hfm.known()
        self.assertEqual(set(names), {repo, f'{repo}/extra'})
        # actually resolved as a side effect -- real types, not None
        self.assertEqual(v2.graph.type_of(f'hf:{repo}'), v2.TABLE)
        self.assertEqual(v2.graph.type_of(f'hf:{repo}/extra'), v2.TABLE)

    def test_known_skips_repos_that_fail_to_expand(self):
        """A tagged repo with no (or a broken) ir_datasets card doesn't break
        discovery of the others -- the tag is a hint, not a guarantee."""
        bad_repo = _repo('untagged-badly')
        good_repo = _repo('goodcard')
        good_card = {'tags': ['ir-datasets'], 'default': 'corpus',
                    'tables': _docs_table('corpus', 'docs'), 'benchmarks': {}}
        def fake_card(repo, commit):
            if repo == bad_repo:
                raise KeyError(f'{repo}: not annotated for ir_datasets')
            return good_card
        self._patch('search', return_value=[bad_repo, good_repo])
        self._patch('_resolve_commit', side_effect=lambda repo, revision: 'deadbeef')
        self._patch('_card', side_effect=fake_card)
        self.assertEqual(hfm.known(), [good_repo])

    def test_known_names_are_qualified_and_opt_in(self):
        """``hf.known_names()`` qualifies whatever the registered ``known()``
        callable returns; it's not part of ``names()``/``list_datasets()``
        unless ``discover=True`` is passed."""
        ctx = mock.patch.object(hfm.hf, '_known_fn', return_value=['owner/a', 'owner/b'])
        ctx.start()
        self.addCleanup(ctx.stop)
        self.assertEqual(hfm.hf.known_names(), {'hf:owner/a', 'hf:owner/b'})
        self.assertNotIn('hf:owner/a', v2.list_datasets())
        self.assertIn('hf:owner/a', v2.list_datasets(discover=True))


if __name__ == '__main__':
    unittest.main()
