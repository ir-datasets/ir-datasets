"""Nodes declare the pip packages they need (see ir_datasets/v2/requirements.py);
loading one without them warns. Every node also records the package it comes from."""
import tomllib
import unittest
import warnings
from pathlib import Path
from unittest import mock

import ir_datasets.v2 as v2
from ir_datasets.v2 import graph, requirements
from ir_datasets.v2.vocabulary import row_for

ROOT = Path(__file__).parent.parent


def _pyproject():
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)


def _scratch():
    """An isolated provider + graph with a Resource -> Table -> Benchmark chain,
    where only the Table declares a requirement."""
    provider = v2.ManifestProvider('xtra', distribution='acme-datasets')
    src = v2.Resource('src', sources=['https://x/y'])
    queries = v2.TsvQueries('queries', source=src, requires=['ijson'])
    plain = v2.TsvQueries('plain', source=src)
    bench = v2.Benchmark('bench', queries=queries)
    provider.register(bench, plain, module='acme.things')   # no installed distribution provides `acme`
    return v2.Graph([provider]), bench, plain


class TestInstallDeclarations(unittest.TestCase):

    def test_core_install_is_small(self):
        core = {line.split('>')[0].strip().lower()
                for line in (ROOT / 'requirements.txt').read_text().splitlines() if line.strip()}
        self.assertEqual({'numpy', 'pyyaml', 'requests', 'tqdm', 'lz4', 'pyarrow'}, core)

    def test_the_all_extra_installs_everything_a_node_can_require(self):
        optional = _pyproject()['project']['optional-dependencies']
        in_all = {requirements.canonical(r.split('>')[0]) for r in optional['all']}
        needed = set()
        for name in graph.providers['irds'].names():
            needed.update(requirements.canonical(r) for r in graph[name].requires)
        self.assertTrue(needed)
        self.assertEqual(set(), needed - in_all)

    def test_every_extra_is_in_all(self):
        optional = _pyproject()['project']['optional-dependencies']
        in_all = {requirements.canonical(r.split('>')[0]) for r in optional['all']}
        for name, reqs in optional.items():
            for r in reqs:
                with self.subTest(extra=name, requirement=r):
                    self.assertIn(requirements.canonical(r.split('>')[0]), in_all)


class TestRequirementNames(unittest.TestCase):

    def test_names_are_deduplicated_by_canonical_form(self):
        self.assertEqual(('huggingface_hub', 'ijson'),
                         requirements.normalize(['huggingface_hub', 'Huggingface-Hub', 'ijson']))

    def test_versions_and_extras_are_rejected(self):
        for bad in ('ijson>=3', 'ijson[x]', '', 'pip install ijson'):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    requirements.normalize([bad])

    def test_a_resource_rejects_a_bad_name(self):
        with self.assertRaises(ValueError):
            v2.Resource('test-req-bad', sources=['https://x/y'], requires=['ijson>=3'])

    def test_install_hint_names_the_packages(self):
        self.assertEqual('pip install ijson lxml', requirements.install_hint(['ijson', 'lxml']))

    def test_missing_uses_installed_distributions(self):
        self.assertEqual(['definitely-not-installed-xyz'],
                         requirements.missing(['requests', 'definitely-not-installed-xyz']))


class TestNodeRequires(unittest.TestCase):

    def test_a_parser_declares_what_its_tables_need(self):
        self.assertEqual(('ijson',), graph['irds:dpr-w100-natural-questions-dev-queries'].requires)

    def test_trec_docs_requirements_follow_the_markup_parser(self):
        src = v2.Resource('test-req-trec-src', sources=['https://x/y'])
        for parser, expected in (('BS4', ('beautifulsoup4', 'lxml')), ('sax', ('lxml',)),
                                 ('tut', ()), ('text', ())):
            with self.subTest(parser=parser):
                self.assertEqual(expected, v2.TrecDocs(f'test-req-trec-{parser}', source=src,
                                                       parser=parser).requires)

    def test_clueweb09_needs_the_warc_fork_and_the_others_need_the_original(self):
        self.assertEqual(('warc3-wet-clueweb09', 'zlib-state'), graph.requires('irds:clueweb09'))
        self.assertEqual(('warc3-wet', 'zlib-state'), graph.requires('irds:clueweb12'))

    def test_own_and_parser_requirements_are_merged(self):
        table = v2.TsvQueries('test-req-merge', source=v2.Resource('test-req-merge-src', sources=['https://x/y']),
                              requires=['ijson'])
        table.parser.requires = ('lxml',)
        self.assertEqual(('ijson', 'lxml'), table.requires)

    def test_effective_requirements_include_what_a_node_is_built_from(self):
        g, bench, plain = _scratch()
        self.assertEqual((), bench.requires)               # declares nothing itself
        self.assertEqual(('ijson',), g.requires('xtra:bench'))
        self.assertEqual((), g.requires('xtra:plain'))
        self.assertEqual((), graph.requires('irds:antique-test'))

    def test_requires_is_reported_at_discovery(self):
        g, _, _ = _scratch()
        found = {(s, p, getattr(o, 'value', o)) for s, p, o in g.providers['xtra'].discover_edges()
                 if p == 'requires-dependency'}
        self.assertEqual({('xtra:queries', 'requires-dependency', 'ijson')}, found)


class TestSourcePackage(unittest.TestCase):

    def test_bundled_nodes_come_from_ir_datasets(self):
        row = row_for(graph['irds:antique-test'])
        self.assertEqual('ir_datasets', row['package'])
        self.assertNotIn('package_version', row)

    def test_the_providers_distribution_is_the_fallback(self):
        g, _, _ = _scratch()
        row = row_for(g['xtra:plain'])
        self.assertEqual('acme-datasets', row['package'])

    def test_a_modules_own_distribution_wins(self):
        with mock.patch.object(requirements, 'distribution_of', return_value='other-pkg'):
            row = row_for(graph['irds:antique-test'])
        self.assertEqual('other-pkg', row['package'])

    def test_package_is_reported_at_discovery(self):
        g, _, _ = _scratch()
        found = {(s, p, getattr(o, 'value', o)) for s, p, o in g.providers['xtra'].discover_edges()
                 if p == 'package'}
        self.assertIn(('xtra:queries', 'package', 'acme-datasets'), found)


class TestLoadWarns(unittest.TestCase):

    def test_warns_when_a_package_is_missing(self):
        g, _, _ = _scratch()
        with mock.patch.object(requirements, 'is_installed', return_value=False):
            with self.assertWarnsRegex(requirements.MissingDependencyWarning,
                                       r"ijson.*pip install ijson"):
                g.load('xtra:bench')

    def test_silent_when_installed(self):
        g, _, _ = _scratch()
        with mock.patch.object(requirements, 'is_installed', return_value=True):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                g.load('xtra:bench')

    def test_silent_for_a_node_that_needs_nothing(self):
        g, _, _ = _scratch()
        with mock.patch.object(requirements, 'is_installed', return_value=False):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                g.load('xtra:plain')
                v2.load('irds:antique-test')

    def test_the_public_load_warns_too(self):
        with mock.patch.object(requirements, 'is_installed', return_value=False):
            with self.assertWarns(requirements.MissingDependencyWarning):
                v2.load('irds:dpr-w100-natural-questions-dev-queries')


if __name__ == '__main__':
    unittest.main()
