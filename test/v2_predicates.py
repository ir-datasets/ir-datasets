"""Predicates: every predicate in the graph is declared (``provider.predicate``)
and qualified by the provider that owns it."""
import re
import unittest
import warnings

import ir_datasets.v2 as v2
from ir_datasets.v2 import graph
from ir_datasets.v2 import vocabulary
from ir_datasets.v2.registry import row_triples


class TestDeclarations(unittest.TestCase):

    def test_one_declaration_covers_edges_and_properties(self):
        everything = v2.graph_properties()
        self.assertEqual({'node', 'literal', 'json', 'type'}, {e['range'] for e in everything.values()})
        self.assertTrue(set(v2.edge_kinds()) <= set(everything))
        self.assertTrue(set(v2.literal_properties()) <= set(everything))
        builtin = {'rdf:type', 'rdfs:subClassOf', 'rdfs:range', 'rdfs:domain', 'rdfs:subPropertyOf'}   # terms, not edges
        self.assertEqual(set(v2.edge_kinds()) | set(v2.literal_properties()) | builtin, set(everything))

    def test_every_predicate_is_prefixed_and_lower_camel_case(self):
        for name in v2.graph_properties():
            with self.subTest(predicate=name):
                prefix, _, local = name.partition(':')
                self.assertTrue(prefix and local)
                self.assertRegex(local, r'^[a-z][A-Za-z0-9]*$')

    def test_only_an_edge_can_be_structural(self):
        with self.assertRaises(ValueError):
            vocabulary.declare_graph_property('acme:x', owner='acme', range='literal', structural=True)

    def test_a_redeclaration_must_agree(self):
        vocabulary.declare_graph_property('acme:agree', owner='acme', range='literal')
        vocabulary.declare_graph_property('acme:agree', owner='acme', range='literal')
        with self.assertRaises(ValueError):
            vocabulary.declare_graph_property('acme:agree', owner='acme', range='json')

    def test_a_provider_declares_only_its_own_predicates(self):
        with self.assertRaises(ValueError):
            v2.ManifestProvider('acme-own').declare_graph_property('irds:foo')


class TestPropertiesInTheGraph(unittest.TestCase):
    """Every graph property is described in the graph itself, in RDFS terms (no OWL)."""

    def setUp(self):
        self.triples = list(graph.providers['irds'].discover_edges())
        self.by_subject = {}
        for s, p, o in self.triples:
            self.by_subject.setdefault(s, []).append((p, getattr(o, 'value', o)))

    def test_each_property_is_an_rdf_property_with_a_range_and_a_comment(self):
        for name, entry in v2.irds.graph_properties.items():
            with self.subTest(property=name):
                facts = self.by_subject[name]
                self.assertIn(('rdf:type', 'rdf:Property'), facts)
                self.assertIn(('rdfs:range', entry.get('range_type') or vocabulary.RANGE_TERMS[entry['range']]), facts)
                if entry.get('domain'):
                    self.assertIn(('rdfs:domain', entry['domain']), facts)
                if entry['desc']:
                    self.assertIn(('rdfs:comment', entry['desc']), facts)

    def test_the_range_says_what_the_values_are(self):
        ranges = {name: dict(self.by_subject[name])['rdfs:range'] for name in
                  ('irds:bundleItem', 'irds:citation', 'irds:column')}
        self.assertEqual({'irds:bundleItem': 'rdfs:Resource', 'irds:citation': 'rdfs:Literal',
                          'irds:column': 'rdf:JSON'}, ranges)

    def test_a_structural_edge_says_so_and_nothing_else_does(self):
        flagged = {s for s, p, o in self.triples if p == 'irds:structuralEdge'}
        structural = {n for n, e in v2.irds.graph_properties.items() if e['structural']}
        self.assertEqual(structural, flagged)
        self.assertTrue({'irds:docs', 'irds:derivedFrom', 'irds:member'} <= flagged)
        self.assertNotIn('irds:bundleItem', flagged)         # membership is informational
        self.assertNotIn('irds:directoryManifest', flagged)

    def test_every_literal_property_of_irds_is_metadata(self):
        for name, entry in v2.irds.graph_properties.items():
            if entry['range'] in ('literal', 'json') and name != 'irds:metadata':
                with self.subTest(property=name):
                    self.assertEqual(('irds:metadata',), tuple(entry['parents']))

    def test_every_property_has_at_most_one_parent(self):
        for name, entry in v2.graph_properties().items():
            with self.subTest(property=name):
                self.assertLessEqual(len(entry['parents']), 1)

    def test_a_suite_is_a_benchmark_and_its_members_are_benchmarks(self):
        self.assertTrue(vocabulary.is_subclass('irds:Suite', 'irds:Benchmark'))
        member = dict(self.by_subject['irds:member'])
        self.assertEqual('irds:Suite', member['rdfs:domain'])
        self.assertEqual('irds:Benchmark', member['rdfs:range'])

    def test_the_ways_a_node_consists_of_parts_are_sub_properties_of_bundle_item(self):
        sub = {s for s, p, o in self.triples if p == 'rdfs:subPropertyOf' and o == 'irds:bundleItem'}
        self.assertEqual({'irds:member', *(f'irds:{e}' for e in v2.ENTITIES)}, sub)
        self.assertNotIn('irds:derivedFrom', sub)         # built from, not a part of

    def test_a_property_says_what_it_applies_to_and_points_at(self):
        docs = dict(self.by_subject['irds:docs'])
        self.assertEqual('irds:Benchmark', docs['rdfs:domain'])
        self.assertEqual('irds:Docs', docs['rdfs:range'])
        self.assertEqual('irds:Bundle', dict(self.by_subject['irds:bundleItem'])['rdfs:domain'])

    def test_a_sub_property_is_no_wider_than_its_parent(self):
        props = v2.graph_properties()
        for name, entry in props.items():
            for parent in entry['parents']:
                if props.get(parent, {}).get('domain') and entry['domain']:
                    with self.subTest(property=name, parent=parent):
                        self.assertTrue(vocabulary.is_subclass(entry['domain'], props[parent]['domain']))

    def test_check_reports_an_edge_from_or_to_the_wrong_class(self):
        acme = v2.ManifestProvider('acme-domain')
        src = v2.Resource('src', sources=['https://example.org/x.tsv'])
        queries = v2.TsvQueries('queries', source=src)
        # a benchmark's docs must be a docs table, a suite's member a bundle
        acme.register(v2.Benchmark('bad', docs=queries, queries=queries),
                      v2.Suite('suite', benchmarks=[]))
        problems = v2.Graph([acme]).check()
        self.assertTrue(any(p.startswith('range: acme-domain:bad --irds:docs--> acme-domain:queries')
                            for p in problems), problems)
        self.assertFalse([p for p in problems if 'acme-domain:suite' in p])

    def test_no_owl(self):
        self.assertFalse(any(str(x).startswith('owl:') for t in self.triples for x in t))

    def test_properties_are_vocabulary_not_nodes(self):
        names = set(graph.list())
        self.assertFalse({'irds:bundleItem', 'irds:citation', 'irds:structuralEdge'} & names)
        self.assertEqual([], graph.check())

    def test_a_provider_describes_only_its_own_properties(self):
        acme = v2.ManifestProvider('acme-describes')
        acme.declare_graph_property('mine', range='literal', desc='x')
        self.assertEqual({'acme-describes:mine'}, {s for s, _, _ in acme.property_triples()})


class TestTheExportedGraph(unittest.TestCase):

    def test_every_exported_predicate_is_declared(self):
        declared = set(v2.graph_properties())
        for prefix in ('irds', 'hf', 'legacy'):
            with self.subTest(provider=prefix):
                with warnings.catch_warnings():
                    warnings.simplefilter('error', vocabulary.UndeclaredPropertyWarning)
                    used = {p for _, p, _ in graph.providers[prefix].discover_edges()}
                self.assertEqual(set(), used - declared)

    def test_types_use_the_rdf_predicates(self):
        triples = list(graph.providers['irds'].discover_edges())
        self.assertIn(('irds:antique-docs', 'rdf:type', 'irds:Docs'), triples)
        self.assertTrue(any(p == 'rdfs:subClassOf' for _, p, _ in triples))

    def test_properties_are_qualified_by_their_declaring_provider(self):
        triples = list(graph.providers['irds'].discover_edges())
        self.assertTrue(any(p == 'irds:license' for _, p, _ in triples))
        self.assertFalse(any(p == 'desc' or p == 'description' for _, p, _ in triples))


class TestFileIntegrity(unittest.TestCase):

    def test_hashes_are_a_property_and_validation_only_names_the_method(self):
        r = v2.Resource('test-pred-hash', sources=['https://x/y'],
                        hash=['md5:' + 'a' * 32, 'sha256:' + 'b' * 64])
        literals = r.discovery_literals()
        self.assertEqual({'type': 'file_hash'}, literals['validation'])
        self.assertEqual(['md5:' + 'a' * 32, 'sha256:' + 'b' * 64], literals['fileHash'])
        triples = list(row_triples('irds:x', {'name': 'irds:x', **literals}))
        self.assertEqual(['md5:' + 'a' * 32, 'sha256:' + 'b' * 64],
                         [o.value for _, p, o in triples if p == 'irds:fileHash'])
        (validation,) = [o for _, p, o in triples if p == 'irds:validation']
        self.assertEqual('{"type": "file_hash"}', validation.value)

    def test_a_resource_without_hashes_has_neither(self):
        literals = v2.Resource('test-pred-nohash', sources=['https://x/y']).discovery_literals()
        self.assertNotIn('fileHash', literals)
        self.assertNotIn('validation', literals)

    def test_a_directory_points_at_its_manifest_with_directory_manifest(self):
        self.assertIn('irds:directoryManifest', v2.edge_kinds())
        self.assertNotIn('irds:validatedBy', v2.graph_properties())
        self.assertEqual('literal', v2.graph_properties()['irds:fileHash']['range'])
        self.assertNotIn('many', v2.graph_properties()['irds:fileHash'])   # every property may repeat
        self.assertTrue(any(k == 'irds:directoryManifest'
                            for name in graph.closure('irds:clueweb12')
                            for k in graph.edges_of(name)))


class TestJsonLayout(unittest.TestCase):

    def test_type_comes_first_then_the_other_keys_sorted(self):
        self.assertEqual('{"type": "url", "order": 0, "url": "https://x/y"}',
                         vocabulary.canonical_json({'url': 'https://x/y', 'order': 0, 'type': 'url'}))
        self.assertEqual('{"type": "directory_manifest", "checks": ["files", "sizes"]}',
                         vocabulary.canonical_json({'checks': ['files', 'sizes'], 'type': 'directory_manifest'}))

    def test_it_applies_at_every_level(self):
        self.assertEqual('{"a": {"type": 1, "b": 2}}',
                         vocabulary.canonical_json({'a': {'b': 2, 'type': 1}}))

    def test_an_object_without_type_is_just_sorted(self):
        self.assertEqual('{"a": 1, "b": 2}', vocabulary.canonical_json({'b': 2, 'a': 1}))

    def test_json_properties_are_serialized_that_way(self):
        (_, _, o), = list(row_triples('irds:x', {'name': 'irds:x', 'validation': {'z': 1, 'type': 'file_hash'}}))
        self.assertEqual('{"type": "file_hash", "z": 1}', o.value)


class TestTableColumnsAndLabels(unittest.TestCase):

    def test_each_column_is_a_document_with_name_order_and_data_type(self):
        columns = graph['irds:antique-docs'].discovery_literals()['column']
        self.assertEqual([{'name': 'doc_id', 'order': 0, 'data_type': 'str'},
                          {'name': 'text', 'order': 1, 'data_type': 'str'}], columns)

    def test_an_optional_field_is_nullable(self):
        columns = graph['irds:nano-beir-fiqa-rcp-qrels'].discovery_literals()['column']
        theta, = [c for c in columns if c['name'] == 'theta']
        self.assertEqual({'name': 'theta', 'order': 3, 'data_type': 'float', 'nullable': True}, theta)
        self.assertNotIn('nullable', [c for c in columns if c['name'] == 'relevance'][0])

    def test_data_types_are_readable_names(self):
        import typing
        from ir_datasets.v2.nodes import _data_type
        self.assertEqual(('int', False), _data_type(int))
        self.assertEqual(('list[str]', False), _data_type(typing.List[str]))
        self.assertEqual(('dict[str, int]', False), _data_type(typing.Dict[str, int]))
        self.assertEqual(('float', True), _data_type(typing.Optional[float]))

    def test_columns_are_graph_triples_not_one_list(self):
        triples = list(row_triples('irds:x', {'name': 'irds:x',
                                              **graph['irds:antique-docs'].discovery_literals()}))
        self.assertEqual(['{"name": "doc_id", "order": 0, "data_type": "str"}',
                          '{"name": "text", "order": 1, "data_type": "str"}'],
                         [o.value for _, p, o in triples if p == 'irds:column'])
        self.assertFalse(any(p in ('irds:columns', 'irds:defs') for _, p, _ in triples))

    def test_each_relevance_label_is_a_document(self):
        labels = graph['irds:antique-train-qrels'].discovery_literals()['relevanceLabel']
        self.assertEqual({1, 2, 3, 4}, {d['label'] for d in labels})
        self.assertTrue(all(set(d) == {'label', 'meaning'} for d in labels))


class TestSources(unittest.TestCase):

    def _described(self, source, order=0):
        import json
        return json.loads(source.describe(order)), source.describe(order)

    def test_a_url_source_is_type_url(self):
        from ir_datasets.v2 import Source
        d = Source('https://example.org/x.tsv').describe(0)
        self.assertEqual({'type': 'url', 'order': 0, 'url': 'https://example.org/x.tsv'}, d)

    def test_google_drive_and_the_mirror_are_just_urls(self):
        from ir_datasets.v2 import Source
        from ir_datasets.v2.sources import describe_sources, _ResolvedMirrorSource
        for url in ('https://drive.google.com/uc?id=abc', 'https://mirror.ir-datasets.com/' + 'a' * 32):
            with self.subTest(url=url):
                self.assertEqual('url', Source(url).describe(0)['type'])
        self.assertEqual('url', _ResolvedMirrorSource('https://mirror.ir-datasets.com/abc').describe(0)['type'])
        text, = describe_sources([Source('https://example.org/x')])
        self.assertTrue(text.startswith('{"type": "url", "order": 0'), text)

    def test_a_manual_source_is_type_manual(self):
        from ir_datasets.v2 import Source
        d = Source.external('some/path', instructions='get it').describe(2)
        self.assertEqual(('manual', 2), (d['type'], d['order']))
        self.assertNotIn('kind', d)


class TestRowTriples(unittest.TestCase):

    def test_an_undeclared_field_is_warned_about_and_left_out(self):
        vocabulary._warned_fields.discard('mystery')
        with self.assertWarnsRegex(vocabulary.UndeclaredPropertyWarning, "'mystery'"):
            triples = list(row_triples('irds:x', {'name': 'irds:x', 'type': 'irds:Docs',
                                                  'mystery': 'v', 'citation': 'd'}))
        predicates = {p for _, p, _ in triples}
        self.assertEqual({'rdf:type', 'irds:citation'}, predicates)

    def test_a_list_is_repeated_triples_for_any_property(self):
        triples = list(row_triples('irds:x', {'name': 'irds:x', 'metric': ['a', 'b'], 'citation': ['d', 'e']}))
        self.assertEqual(['a', 'b'], [o.value for _, p, o in triples if p == 'irds:metric'])
        self.assertEqual(['d', 'e'], [o.value for _, p, o in triples if p == 'irds:citation'])
        self.assertEqual(['d'], [o.value for _, p, o in row_triples('irds:x', {'name': 'irds:x', 'citation': 'd'})])

    def test_a_json_property_is_one_json_literal(self):
        (_, p, o), = [t for t in row_triples('irds:x', {'name': 'irds:x', 'relevanceLabel': {'label': 1, 'meaning': 'relevant'}})]
        self.assertEqual(('irds:relevanceLabel', 'json', '{"label": 1, "meaning": "relevant"}'), (p, o.datatype, o.value))

    def test_a_list_of_json_documents_is_one_literal_per_element(self):
        triples = [t for t in row_triples('irds:x', {'name': 'irds:x', 'source': ['{"a": 1}', {'b': 2}]})]
        self.assertEqual([('{"a": 1}', 'json'), ('{"b": 2}', 'json')],
                         [(o.value, o.datatype) for _, _, o in triples])

    def test_the_owning_providers_own_property_wins(self):
        vocabulary.declare_graph_property('hf:ownedHere', owner='hf', range='literal')
        triples = list(row_triples('hf:x', {'name': 'hf:x', 'ownedHere': 'v'}))
        self.assertEqual([('hf:x', 'hf:ownedHere')], [(s, p) for s, p, _ in triples])


if __name__ == '__main__':
    unittest.main()
