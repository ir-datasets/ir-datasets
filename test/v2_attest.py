"""Tests for ir_datasets.v2.attest: records, hashing, signing, and the ``attest:`` provider.

No server and no network: the provider is given an in-memory stand-in for the client. (The server
has its own tests, with ir-datasets.com/attest.)
"""
import collections
import dataclasses
import enum
import os
import tempfile
import unittest
from typing import NamedTuple
from unittest import mock

import ir_datasets.v2 as v2
from ir_datasets.v2 import vocabulary
from ir_datasets.v2.attest import crypto, hashing, records
from ir_datasets.v2.attest.provider import attest as provider
from ir_datasets.v2.graph import Graph

HASH = {'hashes': ['sha256:' + 'a' * 64], 'source': None}
OK = {'kind': 'hash', 'subject': 'irds:x', 'attested_at': '2026-10-28T12:00:00Z'}


class TestRecords(unittest.TestCase):
    def test_valid(self):
        records.validate('hash', {**OK, **HASH})
        records.validate('hash', {**OK, 'hash_version': 1, 'hashes': ['md5:' + 'b' * 32]})

    def test_invalid_hash_records(self):
        for bad in ({'hashes': ['sha256:abc']}, {'hashes': []}, {'hashes': ['nope:' + 'a' * 8]}, {'hashes': 'x'}):
            with self.assertRaises(ValueError):
                records.validate('hash', {**OK, **bad})

    def test_kind_subject_and_timestamp_are_required(self):
        with self.assertRaises(ValueError):
            records.validate('hash', {**OK, **HASH, 'kind': 'visual'})
        with self.assertRaises(ValueError):
            records.validate('hash', {**OK, **HASH, 'subject': 'antique'})
        with self.assertRaises(ValueError):
            records.validate('hash', {k: v for k, v in {**OK, **HASH}.items() if k != 'attested_at'})

    def test_other_kinds(self):
        with self.assertRaises(ValueError):
            records.validate('evaluation', {**OK, 'kind': 'evaluation', 'system': 'x', 'measures': {'nDCG@10': 'high'}})
        with self.assertRaises(ValueError):
            records.validate('visual', {**OK, 'kind': 'visual', 'verdict': 'maybe'})
        records.validate('evaluation', {**OK, 'kind': 'evaluation', 'system': 'x', 'measures': {'MAP': 0.2}})


class TestTableHash(unittest.TestCase):
    def test_is_content_based(self):
        R = collections.namedtuple('R', 'id text')
        a = hashing.table_hash([R('1', 'x'), R('2', 'y')])
        self.assertEqual(a, hashing.table_hash([{'text': 'x', 'id': '1'}, {'id': '2', 'text': 'y'}]))
        self.assertNotEqual(a, hashing.table_hash([R('2', 'y'), R('1', 'x')]))

    def test_record_types(self):
        class Inner(NamedTuple):
            box: tuple
            css: dict

        class Sub(Inner):      # a subclass of a record type
            pass

        @dataclasses.dataclass
        class DC:
            n: int

        class Color(enum.IntEnum):
            RED = 1
        for value in (Sub((1.0, 2.0), {'b': 'x', 'a': 'y'}), DC(3), Color.RED, b'\x00\xff', {3, 1, 2}, float('nan'), None):
            hashing.table_hash([value])
        self.assertEqual(hashing.table_hash([{1, 2, 3}]), hashing.table_hash([{3, 2, 1}]))
        with self.assertRaises(TypeError):
            hashing.table_hash([object()])

    def test_empty_table(self):
        self.assertTrue(hashing.table_hash([])[:7] == 'sha256:')


class TestSigning(unittest.TestCase):
    def test_roundtrip_and_tamper(self):
        private, public = crypto.generate_keypair()
        record = {**OK, **HASH}
        signature = crypto.sign(private, records.signing_payload(record))
        self.assertTrue(crypto.verify(public, signature, records.signing_payload(record)))
        for changed in ({**record, 'subject': 'irds:other'}, {**record, 'hashes': ['sha256:' + 'b' * 64]}):
            self.assertFalse(crypto.verify(public, signature, records.signing_payload(changed)))
        self.assertFalse(crypto.verify(crypto.generate_keypair()[1], signature, records.signing_payload(record)))
        self.assertFalse(crypto.verify(public, 'not-a-signature', records.signing_payload(record)))

    def test_identity_is_saved_under_the_ir_datasets_home(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {'IR_DATASETS_HOME': home}):
            self.assertIsNone(crypto.Identity.load())
            path = crypto.Identity('alice', *crypto.generate_keypair(), identity_id='abc').save()
            self.assertEqual(os.path.join(home, 'attest', 'identity.json'), str(path))
            self.assertEqual(0o600, os.stat(path).st_mode & 0o777)
            self.assertEqual('alice', crypto.Identity.load().username)


class FakeClient:
    """Stands in for the HTTP client: a list of rows, as the server would return them."""
    url = 'http://fake'

    def __init__(self, rows):
        self.rows, self.asked = rows, []

    def records(self, after=0):
        self.asked.append(after)
        return iter([r for r in self.rows if r['seq'] > after])

    def get(self, *ids):
        return [r for r in self.rows if r['id'] in ids]


def _row(seq, id, kind, subject, record=None, signed_by=None):
    return {'id': id, 'type': kind, 'subject': subject, 'seq': seq, 'verified': bool(signed_by),
            'signature': 'sig' if signed_by else None, 'signed_by': signed_by,
            'record': {'kind': kind, 'subject': subject, **(record or {})}}


class TestProvider(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        patch = mock.patch.dict(os.environ, {'IR_DATASETS_HOME': home.name})
        patch.start()
        os.environ.pop('IR_DATASETS_ATTEST_OFFLINE', None)
        self.addCleanup(patch.stop)
        self.rows = [_row(1, 'id-alice', 'identity', None, {'username': 'alice'}),
                     _row(2, 'id-hash', 'hash', 'irds:antique-docs', HASH, signed_by='id-alice'),
                     _row(3, 'id-eval', 'evaluation', 'irds:antique-test', {'system': 's', 'measures': {'MAP': 1}})]
        self.client = FakeClient(self.rows)
        provider.client = self.client
        self.addCleanup(setattr, provider, 'client', None)
        provider.refresh()
        self.addCleanup(provider.refresh)
        self.graph = Graph([provider, v2.default_graph().providers['irds']])

    def test_types_and_subtypes(self):
        self.assertTrue(vocabulary.is_subclass('attest:HashAttestation', 'attest:Attestation'))
        self.assertEqual('attest:HashAttestation', self.graph.type_of('attest:id-hash'))
        self.assertEqual(['attest:id-eval'], self.graph.list(type='attest:EvaluationAttestation'))
        self.assertEqual(3, len(self.graph.list(type='attest:Attestation')))

    def test_edges(self):
        self.assertEqual(['attest:id-hash'], [n.qualified_name for n in self.graph.targets('irds:antique-docs', 'attest:attestation')])
        self.assertEqual(['attest:id-alice'], [n.qualified_name for n in self.graph.targets('attest:id-hash', 'attest:signedBy')])

    def test_every_type_has_a_json_record(self):
        for id in ('id-alice', 'id-hash', 'id-eval'):
            self.assertIsInstance(self.graph[f'attest:{id}'].record, dict)
        node = self.graph['attest:id-hash']
        self.assertEqual(('irds:antique-docs', 'id-alice', True), (node.subject, node.signed_by, node.verified))
        self.assertEqual(['attest:id-hash'], [n.qualified_name for n in provider.for_node('irds:antique-docs')])

    def test_sync_is_incremental(self):
        self.assertEqual(3, len(provider.rows()))
        self.rows.append(_row(4, 'id-new', 'hash', 'irds:antique-docs', HASH))
        provider.refresh()
        self.assertEqual(4, len(provider.rows()))
        self.assertEqual([0, 3], self.client.asked)       # the second pass asked only for what is newer

    def test_offline_uses_the_copy_on_disk(self):
        provider.rows()
        provider.refresh()
        self.client.rows = None            # would raise if asked
        with mock.patch.dict(os.environ, {'IR_DATASETS_ATTEST_OFFLINE': '1'}):
            self.assertEqual(3, len(provider.rows()))

    def test_unreachable_server_is_not_fatal(self):
        provider.rows()
        provider.refresh()
        self.client.records = mock.Mock(side_effect=OSError('down'))
        with self.assertLogs('ir_datasets.v2.attest.provider', 'WARNING'):
            self.assertEqual(3, len(provider.rows()))

    def test_loading_a_dataset_does_not_discover_attestations(self):
        self.client.asked.clear()
        provider._rows = None
        v2.default_graph().load('irds:antique-test')
        self.assertEqual([], self.client.asked)


if __name__ == '__main__':
    unittest.main()
