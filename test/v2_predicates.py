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
        self.assertEqual(set(v2.edge_kinds()) | set(v2.literal_properties()) | {'rdf:type', 'rdfs:subClassOf'},
                         set(everything))

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
        self.assertTrue(any(p == 'irds:description' for _, p, _ in triples))
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
                                                  'mystery': 'v', 'description': 'd'}))
        predicates = {p for _, p, _ in triples}
        self.assertEqual({'rdf:type', 'irds:description'}, predicates)

    def test_a_list_is_repeated_triples_for_any_property(self):
        triples = list(row_triples('irds:x', {'name': 'irds:x', 'metric': ['a', 'b'], 'description': ['d', 'e']}))
        self.assertEqual(['a', 'b'], [o.value for _, p, o in triples if p == 'irds:metric'])
        self.assertEqual(['d', 'e'], [o.value for _, p, o in triples if p == 'irds:description'])
        self.assertEqual(['d'], [o.value for _, p, o in row_triples('irds:x', {'name': 'irds:x', 'description': 'd'})])

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
