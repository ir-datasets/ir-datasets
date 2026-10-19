"""Unit tests for the ``hf:`` dynamic provider (ir_datasets/v2/hf_provider.py).

No network: every test patches the module's own network-touching functions
(``_resolve_commit``, ``_card``, ``_hf_lib``) with fixtures, matching how the
rest of v2's tests keep structural checks free of downloads. Tests that don't
care about column resolution never trigger it (it's lazy -- see
``_HfFileParser``/``_HfDatasetsParser``); the ones that do also mock
``HfDataset.path``/``_rows_of`` so peeking a table's columns needs no real
file either. Each test uses its own fake repo id so the module-level caches
(``_card_cache``, ``_repo_cache``, ``_node_cache``) and the ``hf`` provider's
registry never collide across tests.
"""
import itertools
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import yaml

import ir_datasets.v2 as v2
from ir_datasets.v2 import hf_provider as hfm

_counter = itertools.count()


def _repo(name):
    """A fresh, never-before-used fake repo id for this test."""
    return f'test-owner/{name}-{next(_counter)}'


def _card_of(*entries, tags=(), meta=None):
    """A parsed card, as ``_card`` would return it from a list of entries."""
    return {'tags': list(tags), 'meta': meta or {}, 'entries': hfm._entries_of('test', list(entries))}


def _table(fragment, type='irds:Docs', **kwargs):
    # no column_map: these are structural tests (existence, identity, naming)
    # that never access .record_type or iterate, so column resolution (which
    # needs I/O) is never triggered.
    return {'fragment': fragment, 'type': type, 'file': f'data/{fragment or "top"}.jsonl', **kwargs}


def _write_readme(text):
    fd, path = tempfile.mkstemp(suffix='.md')
    with open(fd, 'w', encoding='utf-8') as fout:
        fout.write(text)
    return path


class TestV2Hf(unittest.TestCase):
    def setUp(self):
        # every test gets an empty registry (the real one is a directory of the package)
        self.card_dir = Path(tempfile.mkdtemp())
        ctx = mock.patch.object(hfm, 'CARD_DIR', self.card_dir)
        ctx.start()
        self.addCleanup(ctx.stop)

    def _registry_card(self, repo, front):
        path = self.card_dir / f'{repo}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(front))

    def _patch(self, target, **kwargs):
        ctx = mock.patch.object(hfm, target, **kwargs)
        obj = ctx.start()
        self.addCleanup(ctx.stop)
        return obj

    def _mock(self, card_by_repo, commit='deadbeef'):
        """Fixture-backed replacements for every network call the resolver
        makes."""
        self._patch('_resolve_commit', side_effect=lambda repo: commit)
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

    # -- naming ------------------------------------------------------------

    def test_a_fragment_resolves_its_entry(self):
        repo = _repo('single')
        self._mock({repo: _card_of(_table('corpus'))})
        node = v2.graph[f'hf:{repo}/corpus']
        self.assertIsInstance(node, v2.DocTable)
        self.assertEqual(node.entity, 'docs')

    def test_no_fragment_is_the_top_level_entry(self):
        repo = _repo('toplevel')
        self._mock({repo: _card_of(_table(None, 'irds:Docs'), _table('test', 'irds:Queries'))})
        node = v2.graph[f'hf:{repo}']
        self.assertEqual(node.entity, 'docs')
        self.assertEqual(node.qualified_name, f'hf:{repo}')
        self.assertEqual(v2.graph[f'hf:{repo}/test'].entity, 'queries')

    def test_no_top_level_entry_is_a_key_error_even_with_one_entry(self):
        """Nothing is inferred -- not even when there is exactly one entry."""
        repo = _repo('notop')
        self._mock({repo: _card_of(_table('corpus'))})
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}']
        self.assertEqual(v2.graph[f'hf:{repo}/corpus'].entity, 'docs')

    def test_an_undeclared_fragment_is_a_key_error(self):
        repo = _repo('nofrag')
        self._mock({repo: _card_of(_table('corpus'))})
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}/other']

    def test_a_plain_mapping_is_a_one_item_list(self):
        entries = hfm._entries_of('test', {'type': 'irds:Docs', 'file': 'x.jsonl'})
        self.assertEqual([None], list(entries))
        self.assertEqual(['a'], list(hfm._entries_of('test', [{'fragment': 'a', 'type': 'irds:Docs'}])))

    def test_a_fragment_may_contain_slashes(self):
        repo = _repo('slashes')
        self._mock({repo: _card_of({'fragment': 'en/train', 'type': 'irds:Docs'})})
        self.assertEqual(v2.graph[f'hf:{repo}/en/train'].entity, 'docs')

    def test_type_spellings(self):
        self.assertEqual(('table', 'docs'), hfm._type_of('t', {'type': 'irds:Docs'}))
        for given in ('irds:Benchmark', 'irds:AdhocBenchmark', 'irds:QaBenchmark'):
            self.assertEqual(('benchmark', None), hfm._type_of('t', {'type': given}))
        for loose in ('docs', 'Docs', 'irds:docs', 'Benchmark', 'hf:Docs'):   # fully qualified only
            with self.subTest(type=loose), self.assertRaises(ValueError):
                hfm._type_of('t', {'type': loose})

    def test_a_task_type_is_checked_against_the_facets(self):
        repo = _repo('tasktype')
        self._mock({repo: _card_of(
            _table('d'), _table('q', 'irds:Queries'), _table('r', 'irds:Qrels'),
            {'fragment': 'ok', 'type': 'irds:AdhocBenchmark', 'docs': 'd', 'queries': 'q', 'qrels': 'r'},
            {'fragment': 'bad', 'type': 'irds:AdhocBenchmark', 'docs': 'd', 'queries': 'q'},
            {'fragment': 'qa', 'type': 'irds:QaBenchmark', 'queries': 'q'})})
        self.assertEqual('irds:AdhocBenchmark', v2.graph[f'hf:{repo}/ok'].type)
        with self.assertRaisesRegex(KeyError, 'docs, queries and qrels'):
            v2.graph[f'hf:{repo}/bad']
        with self.assertRaisesRegex(KeyError, 'queries and answers'):
            v2.graph[f'hf:{repo}/qa']

    def test_a_bad_card_is_a_value_error(self):
        for bad in ([{'fragment': 'a', 'type': 'irds:Suite'}],      # unsupported type
                    [{'fragment': 'a'}],                            # no type
                    [{'fragment': 'a', 'type': 'irds:Docs'}, {'fragment': 'a', 'type': 'irds:Queries'}],  # duplicate
                    [],                                             # empty
                    ['docs']):                                      # not a mapping
            with self.subTest(card=bad), self.assertRaises(ValueError):
                hfm._entries_of('test', bad)

    # -- tables ------------------------------------------------------------

    def test_no_column_map_takes_every_column(self):
        """No column_map: -> the record type isn't known until the source is
        actually peeked at (needs I/O), so it reflects whatever columns are
        really there -- not assumed to be ir_datasets' minimal (doc_id, text)."""
        repo = _repo('nocolumnmap')
        self._mock({repo: _card_of(_table(None))})
        node = v2.graph[f'hf:{repo}']
        parser = node.parser
        self.assertIsNone(parser.columns)       # not yet resolved -- no I/O has happened
        self._mock_rows([{'doc_id': 'd1', 'title': 'T', 'text': 'hello', 'extra': 'x'}])
        self.assertEqual(node.record_type._fields, ('doc_id', 'title', 'text', 'extra'))
        self.assertEqual(parser.columns, {'doc_id': 'doc_id', 'title': 'title',
                                          'text': 'text', 'extra': 'extra'})

    def test_column_map_renames_and_keeps_unmentioned_columns(self):
        repo = _repo('colmap')
        self._mock({repo: _card_of(_table(None, column_map={'_id': 'doc_id', 'abstract': 'text'}))})
        self._mock_rows([{'_id': 'd1', 'title': 'T', 'abstract': 'hello'}])
        node = v2.graph[f'hf:{repo}']
        self.assertEqual(node.record_type._fields, ('doc_id', 'title', 'text'))
        self.assertEqual(next(iter(node)), node.record_type(doc_id='d1', title='T', text='hello'))

    def test_column_map_null_drops_a_column(self):
        repo = _repo('coldrop')
        self._mock({repo: _card_of(_table(None, column_map={'_id': 'doc_id', 'junk': None}))})
        self._mock_rows([{'_id': 'd1', 'junk': 'ignore-me', 'text': 'hello'}])
        self.assertEqual(v2.graph[f'hf:{repo}'].record_type._fields, ('doc_id', 'text'))

    def test_common_alternative_names_are_mapped_by_type(self):
        f = lambda entity, cols, **kw: hfm._apply_column_map(cols, kw.get('column_map'), entity, kw.get('auto', True))[0]
        self.assertEqual(['doc_id', 'text'], f('docs', ['docno', 'text']))
        self.assertEqual(['doc_id', 'text'], f('docs', ['_id', 'contents']))
        self.assertEqual(['query_id', 'query_unmasked', 'text'], f('queries', ['qid', 'query_unmasked', 'query']))
        self.assertEqual(['query_id', 'doc_id', 'relevance', 'source'],
                         f('qrels', ['qid', 'docno', 'label', 'source']))
        self.assertEqual(['query_id', 'doc_id', 'relevance'], f('qrels', ['query-id', 'corpus-id', 'score']))

    def test_auto_mapping_is_conservative(self):
        f = lambda entity, cols, **kw: hfm._apply_column_map(cols, kw.get('column_map'), entity, kw.get('auto', True))[0]
        # the canonical name is already there: nothing renamed
        self.assertEqual(['doc_id', 'id', 'text'], f('docs', ['doc_id', 'id', 'text']))
        # a title or abstract is never text
        self.assertEqual(['doc_id', 'title', 'abstract'], f('docs', ['docno', 'title', 'abstract']))
        # more than one candidate for the same field is an error, not a guess ...
        with self.assertRaisesRegex(ValueError, "'docno', 'id'.*column_map"):
            f('docs', ['docno', 'id'])
        with self.assertRaises(ValueError):
            f('queries', ['qid', 'query-id', 'text'])
        # ... which naming one in column_map settles
        self.assertEqual(['doc_id', 'id'], f('docs', ['docno', 'id'], column_map={'docno': 'doc_id'}))
        # a column named in column_map is the card's say: left alone (here, mapped elsewhere),
        # so the next alternative takes the canonical name
        self.assertEqual(['key', 'doc_id'], f('docs', ['docno', 'id'], column_map={'docno': 'key'}))
        self.assertEqual(['docno'], f('docs', ['docno'], column_map={'docno': 'docno'}))
        # off, and for an entity with no aliases
        self.assertEqual(['docno', 'text'], f('docs', ['docno', 'text'], auto=False))
        self.assertEqual(['docno'], hfm._apply_column_map(['docno'], None, None)[0])

    def test_auto_mapping_reads_the_source_by_the_original_column(self):
        repo = _repo('automap')
        self._mock({repo: _card_of(_table(None))})
        self._mock_rows([{'docno': 'd1', 'text': 'hello'}])
        node = v2.graph[f'hf:{repo}']
        self.assertEqual(('doc_id', 'text'), node.record_type._fields)
        self.assertEqual('d1', next(iter(node)).doc_id)

    def test_auto_columns_false_takes_the_columns_as_they_are(self):
        repo = _repo('automapoff')
        self._mock({repo: _card_of(_table(None, auto_columns=False))})
        self._mock_rows([{'docno': 'd1', 'text': 'hello'}])
        with self.assertRaises(ValueError):          # no doc_id
            v2.graph[f'hf:{repo}'].record_type

    def test_column_map_unknown_column_is_an_error(self):
        repo = _repo('colunknown')
        self._mock({repo: _card_of(_table(None, column_map={'nope': 'doc_id'}))})
        self._mock_rows([{'_id': 'd1', 'text': 'hello'}])
        with self.assertRaises(ValueError):
            v2.graph[f'hf:{repo}'].record_type

    def test_missing_required_id_field_is_an_error(self):
        repo = _repo('missingid')
        self._mock({repo: _card_of(_table(None))})
        self._mock_rows([{'text': 'hello'}])
        with self.assertRaises(ValueError):
            v2.graph[f'hf:{repo}'].record_type

    def test_no_file_defaults_to_datasets_load_dataset(self):
        repo = _repo('viadatasets')
        self._mock({repo: _card_of({'fragment': 'train', 'type': 'irds:Docs'})})
        node = v2.graph[f'hf:{repo}/train']
        self.assertIsInstance(node.parser, hfm._HfDatasetsParser)
        self.assertEqual((None, 'train', repo), (node.parser.config, node.parser.split, node.parser.repo))

    def test_config_and_split_from_a_compound_fragment(self):
        repo = _repo('viadatasets-cfg')
        self._mock({repo: _card_of({'fragment': 'en/train', 'type': 'irds:Docs'})})
        node = v2.graph[f'hf:{repo}/en/train']
        self.assertEqual(('en', 'train'), (node.parser.config, node.parser.split))

    def test_explicit_split_overrides_the_fragment(self):
        repo = _repo('splitoverride')
        self._mock({repo: _card_of({'fragment': 'docs', 'type': 'irds:Docs', 'split': 'csl'})})
        node = v2.graph[f'hf:{repo}/docs']
        self.assertEqual((None, 'csl'), (node.parser.config, node.parser.split))

    def test_a_top_level_table_without_a_file_needs_a_split(self):
        repo = _repo('topnosplit')
        self._mock({repo: _card_of({'type': 'irds:Docs'})})
        with self.assertRaisesRegex(KeyError, 'needs a file: or a split:'):
            v2.graph[f'hf:{repo}']

    def test_node_properties_pass_through(self):
        repo = _repo('props')
        self._mock({repo: _card_of(_table(None, description='A corpus.', license='cc-by-4.0',
                                          citation='dblp:conf/x/Y', lang='fr'))})
        node = v2.graph[f'hf:{repo}']
        self.assertEqual('fr', node.lang)
        self.assertEqual('cc-by-4.0', node.license)
        self.assertEqual('A corpus.', node.metadata['description'])
        self.assertEqual('dblp:conf/x/Y', node.metadata['citation'])

    def test_the_cards_standard_metadata_fills_in_what_an_entry_leaves_out(self):
        repo = _repo('cardmeta')
        meta = hfm._card_defaults({'license': 'apache-2.0', 'language': ['zh'], 'pretty_name': 'CSL'})
        self.assertEqual({'license': 'apache-2.0', 'lang': 'zh', 'pretty_name': 'CSL'}, meta)
        self._mock({repo: _card_of(_table(None), _table('en', lang='en'),
                                   _table('other', 'irds:Queries', license='cc0-1.0'), meta=meta)})
        top, en, other = (v2.graph[f'hf:{repo}{f}'] for f in ('', '/en', '/other'))
        self.assertEqual(('zh', 'apache-2.0'), (top.lang, top.license))
        self.assertEqual('CSL', top.metadata['prettyName'])
        self.assertEqual(('en', 'apache-2.0'), (en.lang, en.license))       # an entry's own value wins
        self.assertEqual('cc0-1.0', other.license)
        self.assertNotIn('prettyName', en.metadata)                         # the top level only

    def test_the_cards_language_is_for_text_tables_not_qrels(self):
        repo = _repo('langqrels')
        self._mock({repo: _card_of(_table('d'), _table('q', 'irds:Queries'), _table('r', 'irds:Qrels'),
                                   _table('r2', 'irds:Qrels', lang='fr'), meta={'lang': 'en'})})
        self.assertEqual(['en', 'en', None, 'fr'],
                         [v2.graph[f'hf:{repo}/{f}'].lang for f in ('d', 'q', 'r', 'r2')])

    def test_card_defaults_are_conservative(self):
        d = hfm._card_defaults
        self.assertEqual({}, d({'license': 'other', 'language': ['en', 'zh']}))      # ambiguous / unknown
        self.assertEqual({'lang': 'en'}, d({'language': 'en'}))
        self.assertEqual({'license': ['mit', 'apache-2.0']}, d({'license': ['mit', 'apache-2.0']}))

    def test_an_unknown_key_is_reported_and_ignored(self):
        repo = _repo('unknownkey')
        self._mock({repo: _card_of(_table(None, nonsense=1))})
        with mock.patch.object(hfm._logger, 'warn') as warn:
            node = v2.graph[f'hf:{repo}']
        self.assertEqual('docs', node.entity)
        self.assertIn('nonsense', warn.call_args[0][0])

    # -- benchmarks --------------------------------------------------------

    def _benchmark_card(self, top=False, **extra):
        return _card_of(
            _table('corpus'), _table('test', 'irds:Queries'), _table('qrels', 'irds:Qrels'),
            {'fragment': None if top else 'eval', 'type': 'irds:Benchmark', 'docs': 'corpus',
             'queries': 'test', 'qrels': 'qrels', 'metrics': ['nDCG@10'], **extra})

    def test_benchmark_facets_are_the_cards_own_nodes(self):
        repo = _repo('bench')
        self._mock({repo: self._benchmark_card()})
        benchmark = v2.graph[f'hf:{repo}/eval']
        self.assertIsInstance(benchmark, v2.Benchmark)
        self.assertEqual(['nDCG@10'], benchmark.metrics)
        self.assertIs(benchmark.docs, v2.graph[f'hf:{repo}/corpus'])
        self.assertIs(benchmark.queries, v2.graph[f'hf:{repo}/test'])
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}']          # a benchmark entry doesn't make the bare name resolve

    def test_a_top_level_benchmark_is_the_bare_name(self):
        repo = _repo('benchtop')
        self._mock({repo: self._benchmark_card(top=True)})
        node = v2.graph[f'hf:{repo}']
        self.assertIsInstance(node, v2.Benchmark)
        self.assertEqual(f'hf:{repo}', node.qualified_name)
        self.assertIs(v2.graph[f'hf:{repo}/corpus'], node.docs)

    def test_several_benchmarks_share_tables(self):
        repo = _repo('multibench')
        self._mock({repo: _card_of(
            _table('corpus'), _table('train-q', 'irds:Queries'), _table('test-q', 'irds:Queries'),
            {'fragment': 'train', 'type': 'irds:Benchmark', 'docs': 'corpus', 'queries': 'train-q'},
            {'fragment': 'test', 'type': 'irds:Benchmark', 'docs': 'corpus', 'queries': 'test-q'})})
        train, test = v2.graph[f'hf:{repo}/train'], v2.graph[f'hf:{repo}/test']
        self.assertIsNot(train, test)
        self.assertIs(train.docs, test.docs)

    def test_a_facet_can_borrow_a_node_from_another_repo(self):
        """A qualified facet is resolved lazily against the graph, exactly like
        ``Benchmark``'s own name-string facets -- here another repo's top level."""
        corpus_repo, eval_repo = _repo('corpus'), _repo('borrower')
        self._mock({
            corpus_repo: _card_of(_table(None)),
            eval_repo: _card_of(_table('test-q', 'irds:Queries'),
                                {'fragment': 'eval', 'type': 'irds:Benchmark',
                                 'docs': f'hf:{corpus_repo}', 'queries': 'test-q'})})
        benchmark = v2.graph[f'hf:{eval_repo}/eval']
        self.assertEqual(f'hf:{corpus_repo}', benchmark.docs.qualified_name)
        self.assertIs(benchmark.docs, v2.graph[f'hf:{corpus_repo}'])

    # -- bundles -----------------------------------------------------------

    def test_a_bundle_groups_the_other_entries(self):
        repo = _repo('bundle')
        self._mock({repo: _card_of(
            {'type': 'irds:Bundle', 'summary': '# All of it', 'description': 'short'},
            _table('corpus'), _table('test', 'irds:Queries'),
            {'fragment': 'eval', 'type': 'irds:Benchmark', 'docs': 'corpus', 'queries': 'test'})})
        bundle = v2.graph[f'hf:{repo}']
        self.assertEqual('irds:Bundle', bundle.type)
        self.assertEqual(f'hf:{repo}', bundle.qualified_name)
        self.assertEqual('# All of it', bundle.metadata['summary'])
        self.assertEqual([f'hf:{repo}/corpus', f'hf:{repo}/test', f'hf:{repo}/eval'],
                         [p.qualified_name for p in bundle.parts])
        self.assertEqual(['irds:hasPart'] * 3, [e.kind for e in bundle.structural_edges()])

    def test_an_overview_is_a_bundle_with_a_required_summary(self):
        repo = _repo('overview')
        self._mock({repo: _card_of(
            {'type': 'irds:Overview', 'summary': 'The whole story.'}, _table('corpus'),
            {'fragment': 'nosummary', 'type': 'irds:Overview', 'parts': ['corpus']})})
        overview = v2.graph[f'hf:{repo}']
        self.assertEqual('irds:Overview', overview.type)
        self.assertEqual('The whole story.', overview.metadata['summary'])
        with self.assertRaisesRegex(KeyError, 'needs a summary'):
            v2.graph[f'hf:{repo}/nosummary']

    def test_a_bundle_can_name_its_parts_and_borrow_from_elsewhere(self):
        repo, other = _repo('bundleparts'), _repo('bundleelsewhere')
        self._mock({repo: _card_of(
            {'fragment': 'all', 'type': 'irds:Bundle', 'parts': ['corpus', f'hf:{other}']},
            _table('corpus'), _table('unlisted')),
            other: _card_of(_table(None))})
        bundle = v2.graph[f'hf:{repo}/all']
        v2.graph[f'hf:{other}']     # a borrowed part is a node of its own: load it, so the graph is whole
        self.assertEqual([f'hf:{repo}/corpus', f'hf:{other}'],
                         [p if isinstance(p, str) else p.qualified_name for p in bundle.parts])

    def test_bundles_do_not_include_each_other_by_default_and_cycles_are_errors(self):
        repo = _repo('bundlecycle')
        self._mock({repo: _card_of(
            {'type': 'irds:Bundle'}, {'fragment': 'inner', 'type': 'irds:Bundle'}, _table('corpus'))})
        self.assertEqual([f'hf:{repo}/corpus'],
                         [p.qualified_name for p in v2.graph[f'hf:{repo}'].parts])
        repo = _repo('bundlecycle2')
        self._mock({repo: _card_of(
            {'fragment': 'a', 'type': 'irds:Bundle', 'parts': ['b']},
            {'fragment': 'b', 'type': 'irds:Bundle', 'parts': ['a']})})
        with self.assertRaisesRegex(KeyError, 'cycle'):
            v2.graph[f'hf:{repo}/a']

    def test_a_bundle_with_nothing_to_bundle_is_an_error(self):
        repo = _repo('bundleempty')
        self._mock({repo: _card_of({'type': 'irds:Bundle'})})
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}']

    # -- the card, as stored -----------------------------------------------

    def test_both_card_keys_present_is_an_error(self):
        repo = _repo('dupe')
        readme = ('---\n' + yaml.safe_dump({'ir_datasets': {'type': 'irds:Docs'},
                                            'ir-datasets': {'type': 'irds:Docs'}}) + '---\nbody\n')
        self._patch('_hf_lib')
        hfm._hf_lib().hf_hub_download.return_value = _write_readme(readme)
        with self.assertRaises(ValueError):
            hfm._card(repo, 'deadbeef')

    def test_unannotated_repo_is_a_key_error(self):
        repo = _repo('bare')
        self._patch('_hf_lib')
        hfm._hf_lib().hf_hub_download.return_value = _write_readme('---\nlicense: mit\n---\nbody\n')
        with self.assertRaises(KeyError):
            hfm._card(repo, 'deadbeef')

    def test_a_card_in_yaml_form(self):
        """The list form and the single-mapping shorthand, parsed from real frontmatter."""
        self._patch('_hf_lib')
        for block in ([{'fragment': None, 'type': 'irds:Docs', 'file': 'x.jsonl'},
                       {'fragment': 'q', 'type': 'irds:Queries', 'file': 'q.jsonl'}],
                      {'type': 'irds:Docs', 'file': 'x.jsonl'}):
            with self.subTest(block=block):
                repo = _repo('yaml')
                hfm._hf_lib().hf_hub_download.return_value = _write_readme(
                    '---\n' + yaml.safe_dump({'ir-datasets': block}) + '---\nbody\n')
                card = hfm._card(repo, 'deadbeef')
                self.assertIn(None, card['entries'])

    # -- the registry ------------------------------------------------------

    def test_a_registry_card_needs_no_network_and_no_readme(self):
        repo = _repo('registry')
        sha = 'a' * 40
        self._registry_card(repo, {
            'license': 'cc-by-4.0', 'language': ['en'], 'commit': sha,
            'ir-datasets': [{'type': 'irds:Docs', 'file': 'docs.jsonl'},
                            {'fragment': 'q', 'type': 'irds:Queries', 'file': 'q.jsonl'}]})
        self._patch('_resolve_commit', side_effect=AssertionError('the commit is pinned'))
        self._patch('_hf_lib', side_effect=AssertionError('no README is fetched'))
        docs = v2.graph[f'hf:{repo}']
        self.assertEqual(('irds:Docs', 'en', 'cc-by-4.0'), (docs.type, docs.lang, docs.license))
        self.assertEqual('irds:Queries', v2.graph[f'hf:{repo}/q'].type)
        git = self._resource_of(docs)
        self.assertEqual(sha, git.commit)

    def test_the_registry_wins_over_the_repos_own_card(self):
        repo = _repo('registrywins')
        self._registry_card(repo, {'ir-datasets': {'type': 'irds:Queries', 'file': 'q.jsonl'}})
        self._patch('_resolve_commit', return_value='b' * 40)
        self._patch('_hf_lib', side_effect=AssertionError('the registry card is used'))
        self.assertEqual('irds:Queries', v2.graph[f'hf:{repo}'].type)

    def test_without_a_registry_card_the_readme_is_used(self):
        repo = _repo('noregistry')
        self._mock({repo: _card_of(_table(None))})
        self.assertEqual('irds:Docs', v2.graph[f'hf:{repo}'].type)

    def test_a_registry_commit_must_be_a_full_hash(self):
        repo = _repo('badpin')
        self._registry_card(repo, {'commit': 'main', 'ir-datasets': {'type': 'irds:Docs', 'file': 'x'}})
        with self.assertRaisesRegex(KeyError, '40-character'):
            v2.graph[f'hf:{repo}']

    def test_a_malformed_registry_card_is_an_error(self):
        repo = _repo('malformed')
        path = self.card_dir / f'{repo}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('- just\n- a list\n')
        with self.assertRaises(ValueError):
            hfm._registry_front(repo)

    def test_known_lists_the_registry_as_well_as_the_tagged_repos(self):
        in_registry, tagged = _repo('listed'), _repo('tagged')
        self._registry_card(in_registry, {'commit': 'c' * 40,
                                          'ir-datasets': {'type': 'irds:Docs', 'file': 'x.jsonl'}})
        self._patch('search', return_value=[tagged])
        self._patch('_resolve_commit', side_effect=lambda repo: 'deadbeef')
        self._patch('_card', side_effect=lambda repo, commit: (
            hfm._card_of_front(repo, hfm._registry_front(repo)) if repo == in_registry
            else _card_of(_table(None), tags=['ir-datasets'])))
        self.assertEqual({in_registry, tagged}, set(hfm.known()))

    # -- identity ----------------------------------------------------------

    def test_a_revision_in_the_name_is_not_supported(self):
        repo = _repo('rev')
        self._mock({repo: _card_of(_table('corpus'))})
        with self.assertRaises(KeyError):
            v2.graph[f'hf:{repo}@v1/corpus']

    def _resource_of(self, table):
        source = table.source
        return source._parent if hasattr(source, '_parent') else source

    def test_shared_hf_dataset_across_entries(self):
        repo = _repo('reuse')
        self._mock({repo: _card_of(_table('corpus'), _table('test', 'irds:Queries'))})
        docs, queries = v2.graph[f'hf:{repo}/corpus'], v2.graph[f'hf:{repo}/test']
        self.assertIs(self._resource_of(docs), self._resource_of(queries))
        self.assertIsInstance(self._resource_of(docs), hfm.HfDataset)

    def test_graph_check_clean(self):
        repo = _repo('checkclean')
        self._mock({repo: self._benchmark_card(top=True)})
        self.assertIsInstance(v2.graph[f'hf:{repo}'], v2.Benchmark)
        self.assertEqual(v2.graph.check(), [])

    def test_hf_dataset_name_disambiguates_from_table(self):
        repo = _repo('gitname')
        self._mock({repo: _card_of(_table(None))})
        table = v2.graph[f'hf:{repo}']
        hf_dataset = self._resource_of(table)
        self.assertEqual(table.qualified_name, f'hf:{repo}')
        self.assertEqual(hf_dataset.qualified_name, f'hf:{repo}.git')

    # -- discovery ---------------------------------------------------------

    def test_discovery_reports_what_the_card_says_without_touching_the_data(self):
        repo = _repo('literals')
        self._mock({repo: _card_of(_table(None, lang='fr', license='mit', description='d'))})
        self._patch('known')
        with mock.patch.object(hfm.HfDataset, 'path', side_effect=AssertionError('read the data')), \
                mock.patch.object(hfm, '_rows_of', side_effect=AssertionError('read the data')):
            v2.graph[f'hf:{repo}']
            triples = {(s, p): o for s, p, o in hfm.hf.discover_edges()}
        name = f'hf:{repo}'
        self.assertEqual('fr', triples[(name, 'irds:lang')].value)
        self.assertEqual('mit', triples[(name, 'irds:license')].value)
        self.assertEqual('d', triples[(name, 'irds:description')].value)
        self.assertFalse(any(p == 'hf:snapshotAt' for _, p in triples))
        self.assertFalse(any(p == 'irds:column' for _, p in triples))   # only the data can say

    def test_search_lists_hub_repos_tagged_ir_datasets(self):
        fake_hub = mock.Mock()
        fake_hub.HfApi.return_value.list_datasets.return_value = [
            mock.Mock(id='owner/a'), mock.Mock(id='owner/b')]
        self._patch('_hf_lib', return_value=fake_hub)
        self.assertEqual(hfm.search(None), ['owner/a', 'owner/b'])
        fake_hub.HfApi.return_value.list_datasets.assert_called_once_with(
            search=None, filter=hfm.TAG)

    def test_known_expands_each_tagged_repo_into_its_addressable_names(self):
        repo = _repo('discoverable')
        self._patch('search', return_value=[repo])
        self._mock({repo: _card_of(_table(None), _table('extra', 'irds:Queries'), tags=['ir-datasets'])})
        self.assertEqual(set(hfm.known()), {repo, f'{repo}/extra'})
        # (the graph's own catalog is cached per provider, so read the nodes' types directly)
        self.assertEqual('irds:Docs', v2.graph[f'hf:{repo}'].type)
        self.assertEqual('irds:Queries', v2.graph[f'hf:{repo}/extra'].type)

    def test_known_skips_repos_that_fail_to_expand(self):
        bad_repo, good_repo = _repo('untagged-badly'), _repo('goodcard')
        good_card = _card_of(_table(None), tags=['ir-datasets'])
        def fake_card(repo, commit):
            if repo == bad_repo:
                raise KeyError(f'{repo}: not annotated for ir_datasets')
            return good_card
        self._patch('search', return_value=[bad_repo, good_repo])
        self._patch('_resolve_commit', side_effect=lambda repo: 'deadbeef')
        self._patch('_card', side_effect=fake_card)
        self.assertEqual(hfm.known(), [good_repo])


if __name__ == '__main__':
    unittest.main()
