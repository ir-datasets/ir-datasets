import io
import os
import tarfile
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from ir_datasets.v2.base import Literal
from ir_datasets.v2 import documentation_provider as dp
from ir_datasets.v2.documentation_provider import DocumentationProvider, link_targets, parse


def provider_with(pages):
    tmp = tempfile.TemporaryDirectory()
    for rel, text in pages.items():
        path = Path(tmp.name) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    provider = DocumentationProvider(directory=tmp.name)
    provider._tmp = tmp
    return provider


PAGE = '---\ndescription: "A: thing."\n---\nBody [x](irds:antique-test) and [p](dblp:conf/a/B) and [w](https://e.org/#a).\n'


class TestDocumentation(unittest.TestCase):
    def test_parse(self):
        meta, body = parse(PAGE)
        self.assertEqual(meta, {'description': 'A: thing.'})
        self.assertTrue(body.startswith('Body'))
        self.assertEqual(parse('no frontmatter')[0], {})

    def test_link_targets_skip_fences_and_images(self):
        body = '[a](irds:x)\n```\n[b](irds:y)\n```\n![i](irds:z)\n'
        self.assertEqual(link_targets(body), ['irds:x'])

    def test_triples(self):
        p = provider_with({'irds/thing.md': PAGE})
        triples = {(s, k): o for s, k, o in p.discover_edges() if s == 'irds:thing'}
        self.assertEqual(triples[('irds:thing', 'documentation:description')].value, 'A: thing.')
        self.assertIsInstance(triples[('irds:thing', 'documentation:about')], Literal)
        # only graph links are edges: dblp/https are not
        self.assertEqual([o for s, k, o in p.discover_edges() if k == 'documentation:links'], ['irds:antique-test'])

    def test_nested_names(self):
        p = provider_with({'hf/owner/repo.md': PAGE})
        self.assertEqual(list(p.pages()), ['hf:owner/repo'])

    def test_lint(self):
        p = provider_with({'irds/a.md': '---\ncolor: red\n---\n[r](rel) [s](irds:x#sec) [u](nope:y)\n'})
        problems = ' '.join(p.lint(known_prefixes={'irds', 'documentation'}))
        for expected in ("unknown frontmatter key 'color'", 'no description', 'relative link',
                         'section anchors', "unknown scheme 'nope'"):
            self.assertIn(expected, problems)

    def test_bad_page_is_skipped_not_fatal(self):
        p = provider_with({'irds/bad.md': '---\ndescription: a: b\n---\nx', 'irds/ok.md': PAGE})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            names = {s for s, _, _ in p.discover_edges() if s.startswith('irds:')}
        self.assertEqual(names, {'irds:ok'})
        self.assertTrue(any('bad.md' in str(w.message) for w in caught))
        self.assertTrue(any('unreadable' in x for x in p.lint(known_prefixes={'irds'})))

    def test_no_directory_is_empty(self):
        self.assertEqual(list(DocumentationProvider(directory='/nonexistent/x').discover_edges())[0][1], 'rdf:type')


class TestIndex(unittest.TestCase):
    def values(self, provider, name):
        return {k.split(':')[1]: o.value for s, k, o in provider.discover_edges()
                if s == name and isinstance(o, Literal) and k != 'documentation:source'}

    def source(self, provider, name):
        (value,) = [o.value for s, k, o in provider.discover_edges()
                    if s == name and k == 'documentation:source']
        return value

    def test_a_node_reports_the_one_file_it_was_read_from(self):
        p = provider_with({'irds/_index.yaml': 'thing: idx\nonly: just idx\n',
                           'irds/thing.md': '---\ndescription: page\n---\nBody.\n'})
        self.assertEqual(self.source(p, 'irds:thing'), 'irds/thing.md')
        self.assertEqual(self.source(p, 'irds:only'), 'irds/_index.yaml')

    def test_a_page_with_only_a_description_is_allowed(self):
        p = provider_with({'irds/thing.md': '---\ndescription: x\n---\n'})
        self.assertEqual(self.values(p, 'irds:thing'), {'description': 'x'})
        self.assertFalse(p.lint(known_prefixes={'irds'}))

    def test_entry_is_a_description_or_a_mapping_and_has_no_page(self):
        p = provider_with({'irds/_index.yaml': 'a: "Short: one."\nb:\n  description: Two.\n'})
        self.assertEqual(self.values(p, 'irds:a'), {'description': 'Short: one.'})
        self.assertEqual(self.values(p, 'irds:b'), {'description': 'Two.'})

    def test_a_page_replaces_the_index_entry_outright(self):
        p = provider_with({'irds/_index.yaml': 'thing: from index\n',
                           'irds/thing.md': PAGE})
        values = self.values(p, 'irds:thing')
        self.assertEqual(values['description'], 'A: thing.')
        self.assertIn('about', values)
        self.assertIn('has both', ' '.join(p.lint(known_prefixes={'irds'})))

    def test_a_page_needs_its_own_description_even_if_the_index_has_one(self):
        p = provider_with({'irds/_index.yaml': 'thing: Indexed.\n', 'irds/thing.md': 'Body only.\n'})
        self.assertNotIn('description', self.values(p, 'irds:thing'))
        self.assertIn('no description', ' '.join(p.lint(known_prefixes={'irds'})))

    def test_an_unreadable_page_falls_back_to_the_index_entry(self):
        p = provider_with({'irds/_index.yaml': 'thing: Indexed.\n', 'irds/thing.md': '---\ndescription: a: b\n---\nx'})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            self.assertEqual(self.values(p, 'irds:thing'), {'description': 'Indexed.'})
            self.assertEqual(self.source(p, 'irds:thing'), 'irds/_index.yaml')

    def test_lint(self):
        p = provider_with({'irds/_index.yaml': 'a:\n  colour: red\nb: ""\nthing: x\n', 'irds/thing.md': PAGE})
        problems = ' '.join(p.lint(known_prefixes={'irds'}))
        for expected in ("irds:a: unknown index key 'colour'", 'irds:a: index entry has no description',
                         'irds:b: index entry has no description', 'irds:thing: has both'):
            self.assertIn(expected, problems)

    def test_bad_index_is_skipped_not_fatal(self):
        p = provider_with({'irds/_index.yaml': '- not\n- a mapping\n', 'hf/_index.yaml': 'x: ok\n'})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(self.values(p, 'hf:x'), {'description': 'ok'})
        self.assertTrue(any('irds' in str(w.message) for w in caught))
        self.assertTrue(any('unreadable index' in x for x in p.lint(known_prefixes={'irds'})))


class TestVocabulary(unittest.TestCase):
    def test_every_documentation_property_is_under_the_root(self):
        from ir_datasets.v2.vocabulary import graph_properties
        props = {k: v for k, v in graph_properties().items() if k.startswith('documentation:')}
        root = 'documentation:documentation'
        self.assertEqual(props[root]['parents'], ('irds:metadata',))
        for name, entry in props.items():
            if name != root:
                self.assertIn(root, entry['parents'], name)
        self.assertIn('documentation:source', props)
        self.assertFalse([k for k in props if k.endswith('Source')])


class TestEditLinks(unittest.TestCase):
    def test_edit_url_targets_the_branch_of_the_configured_repo(self):
        self.assertEqual(dp.edit_url('irds/mast.md'), 'https://github.com/ir-datasets/documentation/edit/main/irds/mast.md')
        with mock.patch.dict(os.environ, {dp.ENV_REPO: 'me/fork'}):
            self.assertEqual(dp.edit_url('irds/_index.yaml'), 'https://github.com/me/fork/edit/main/irds/_index.yaml')

    def test_new_page_url_prefills_the_name_and_a_skeleton(self):
        from urllib.parse import parse_qs, urlparse
        url = dp.new_page_url('irds:mast', description='A: b.', parts=['irds:x', 'irds:y'])
        parsed = urlparse(url)
        self.assertEqual(parsed.path, '/ir-datasets/documentation/new/main/irds')
        query = parse_qs(parsed.query)
        self.assertEqual(query['filename'], ['mast.md'])
        meta, body = parse(query['value'][0])
        self.assertEqual(meta, {'description': 'A: b.'})
        self.assertIn('[irds:x](irds:x)', body)

    def test_index_path_and_line(self):
        self.assertEqual(dp.index_path('irds:antique-test'), 'irds/_index.yaml')
        self.assertEqual(dp.index_line('irds:antique-test'), 'antique-test: "One sentence saying what this is."')
        self.assertEqual(dp.index_line('hf:owner/repo', 'A: b'), 'hf:owner/repo'.split(':')[1] + ': "A: b"')
        self.assertEqual(dp.index_line('irds:we ird'), '"we ird": "One sentence saying what this is."')
        # what it produces is a valid index entry
        meta = dp.DocumentationProvider._read_index
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / '_index.yaml').write_text(dp.index_line('irds:antique-test', 'A: b') + '\n')
            self.assertEqual(meta(Path(d) / '_index.yaml'), {'antique-test': {'description': 'A: b'}})

    def test_nested_names_go_in_nested_directories_and_long_templates_are_capped(self):
        from urllib.parse import parse_qs, urlparse
        self.assertEqual(urlparse(dp.new_page_url('hf:owner/repo')).path, '/ir-datasets/documentation/new/main/hf/owner')
        self.assertEqual(parse_qs(urlparse(dp.new_page_url('hf:owner/repo')).query)['filename'], ['repo.md'])
        long = dp.template('irds:x', description='d' * 5000)
        self.assertLessEqual(len(long), dp.MAX_TEMPLATE_CHARS + 1)


def tarball(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f'documentation-main/{name}')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Response:
    def __init__(self, status, content=b'', etag=None):
        self.status_code, self.content, self.headers = status, content, {'ETag': etag} if etag else {}


class TestLiveFetch(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        patch = mock.patch.dict(os.environ, {'IR_DATASETS_HOME': self.home.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.home.cleanup)

    def get(self, *responses):
        return mock.patch('requests.get', side_effect=list(responses))

    def test_fetch_extracts_only_markdown_inside_the_directory(self):
        data = tarball({'irds/a.md': b'x', 'irds/_index.yaml': b'a: b', 'LICENSE': b'l', 'irds/../../evil.md': b'e',
                        'notes.txt': b'n', 'irds/other.yaml': b'z: 1'})
        with self.get(Response(200, data, '"v1"')):
            root = dp.fetch(ttl=0)
        self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()), ['irds/_index.yaml', 'irds/a.md'])

    def test_fresh_cache_makes_no_request_then_revalidates_with_etag(self):
        with self.get(Response(200, tarball({'irds/a.md': b'v1'}), '"v1"')):
            dp.fetch(ttl=0)
        with self.get() as get:                       # fresh: no request at all
            dp.fetch(ttl=3600)
            get.assert_not_called()
        with self.get(Response(304)) as get:          # stale: conditional, unchanged
            root = dp.fetch(ttl=0)
            self.assertEqual(get.call_args.kwargs['headers'], {'If-None-Match': '"v1"'})
        self.assertEqual((root / 'irds/a.md').read_bytes(), b'v1')
        with self.get(Response(200, tarball({'irds/a.md': b'v2'}), '"v2"')):   # stale: changed
            root = dp.fetch(ttl=0)
        self.assertEqual((root / 'irds/a.md').read_bytes(), b'v2')

    def test_offline_uses_cache_with_a_warning_and_no_cache_means_none(self):
        with self.get(ConnectionError('offline')), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertIsNone(dp.fetch(ttl=0))
        self.assertTrue(caught)
        with self.get(Response(200, tarball({'irds/a.md': b'v1'}), '"v1"')):
            dp.fetch(ttl=0)
        with self.get(ConnectionError('offline')), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertIsNotNone(dp.fetch(ttl=0))
        self.assertIn('cached copy', str(caught[0].message))

    def test_ttl_comes_from_the_environment_and_refresh_ignores_it(self):
        with self.get(Response(200, tarball({'irds/a.md': b'v1'}), '"v1"')):
            dp.fetch(ttl=0)
        with self.get() as get, mock.patch.dict(os.environ, {dp.ENV_TTL: '3600'}):
            dp.fetch()
            get.assert_not_called()
        with self.get(Response(200, tarball({'irds/a.md': b'v2'}), '"v2"')) as get, \
                mock.patch.dict(os.environ, {dp.ENV_TTL: '3600'}):
            root = dp.refresh()                  # a recent edit must show up despite a long TTL
            self.assertEqual(get.call_count, 1)
        self.assertEqual((root / 'irds/a.md').read_bytes(), b'v2')

    def test_full_sha_is_fetched_once_and_never_refreshed(self):
        sha = 'a' * 40
        with self.get(Response(200, tarball({'irds/a.md': b'v1'}), '"v1"')):
            dp.fetch(ref=sha, ttl=0)
        with self.get() as get:
            dp.fetch(ref=sha, ttl=0)
            get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
