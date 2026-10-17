"""Unit tests for ``ir_datasets/v2/log_utils.py`` -- the generic,
provider-agnostic download-event log every ``nodes.Resource`` writes to on
an actual fetch (see ``nodes.Resource.path``/``.stream``).

No network: a fake ``Source`` serves fixed bytes directly, the same pattern
``test/v2_trec_browser.py`` uses for its own soft-hash tests.
"""
import contextlib
import gzip
import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from ir_datasets.util.download import BaseDownload
from ir_datasets.v2 import log_utils
from ir_datasets.v2.nodes import Resource
from ir_datasets.v2.sources import Source


class _FixedBytesSource(Source):
    """A ``Source`` that always serves ``content``, bypassing any network."""

    def __init__(self, content):
        super().__init__('https://example.invalid/fixed-bytes')
        self._content = content

    def _build(self, file):
        content = self._content

        class _Download(BaseDownload):
            def stream(self):
                return contextlib.nullcontext(io.BytesIO(content))

        return _Download()


@contextlib.contextmanager
def _isolated_home():
    """A throwaway ``IR_DATASETS_HOME``, so these tests never touch the
    real cache or the real download log."""
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.dict(os.environ, {'IR_DATASETS_HOME': tmp}):
            yield tmp


def _read_log_entries():
    path = log_utils.log_path()
    if not path.exists():
        return []
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestLogDownload(unittest.TestCase):
    def test_log_download_appends_one_jsonl_entry(self):
        with _isolated_home():
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'some-file')
                content = b'hello world'
                with open(path, 'wb') as fh:
                    fh.write(content)

                class _Node:
                    name = 'test-node'
                    qualified_name = None

                log_utils.log_download(_Node(), path)
                entries = _read_log_entries()
        self.assertEqual(1, len(entries))
        entry = entries[0]
        self.assertEqual('test-node', entry['node'])
        self.assertEqual(hashlib.md5(content).hexdigest(), entry['md5'])
        self.assertEqual(path, entry['path'])
        self.assertIn('timestamp', entry)

    def test_log_download_prefers_qualified_name(self):
        with _isolated_home():
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'f')
                with open(path, 'wb') as fh:
                    fh.write(b'x')

                class _Node:
                    name = 'bare-name'
                    qualified_name = 'irds:provider/bare-name'

                log_utils.log_download(_Node(), path)
                entries = _read_log_entries()
        self.assertEqual('irds:provider/bare-name', entries[0]['node'])

    def test_log_download_is_best_effort_on_a_missing_file(self):
        with _isolated_home():
            class _Node:
                name = 'missing'
                qualified_name = None

            # Must not raise -- a logging failure can never break a download
            # that already succeeded.
            log_utils.log_download(_Node(), '/no/such/path')
            self.assertEqual([], _read_log_entries())

    def test_multiple_downloads_append_rather_than_overwrite(self):
        with _isolated_home():
            with tempfile.TemporaryDirectory() as d:
                for i in range(3):
                    path = os.path.join(d, f'f{i}')
                    with open(path, 'wb') as fh:
                        fh.write(f'content-{i}'.encode('utf8'))

                    class _Node:
                        name = f'node-{i}'
                        qualified_name = None
                    log_utils.log_download(_Node(), path)
                entries = _read_log_entries()
        self.assertEqual(['node-0', 'node-1', 'node-2'],
                          [e['node'] for e in entries])


class TestResourceDownloadLogging(unittest.TestCase):
    def test_a_fresh_download_via_stream_logs_once(self):
        content = b'some bytes to download'
        with _isolated_home():
            resource = Resource('test-log-stream', sources=[_FixedBytesSource(content)])
            with resource.stream() as f:
                self.assertEqual(content, f.read())
            entries = _read_log_entries()
            self.assertEqual(1, len(entries))
            self.assertEqual('test-log-stream', entries[0]['node'])
            self.assertEqual(hashlib.md5(content).hexdigest(), entries[0]['md5'])
            self.assertTrue(os.path.exists(entries[0]['path']))

    def test_a_cache_hit_via_stream_does_not_log_again(self):
        content = b'some bytes to download'
        with _isolated_home():
            resource = Resource('test-log-stream-cached', sources=[_FixedBytesSource(content)])
            with resource.stream() as f:
                f.read()
            with resource.stream() as f:
                # Second read: already on disk, no fetch -- no new source
                # is even consulted, so this would error if it tried.
                self.assertEqual(content, f.read())
            entries = _read_log_entries()
        self.assertEqual(1, len(entries))

    def test_a_fresh_download_via_path_logs_once(self):
        content = b'some other bytes'
        with _isolated_home():
            resource = Resource('test-log-path', sources=[_FixedBytesSource(content)])
            p = resource.path()
            entries = _read_log_entries()
        self.assertEqual(1, len(entries))
        self.assertEqual('test-log-path', entries[0]['node'])
        self.assertEqual(p, entries[0]['path'])

    def test_path_with_force_false_and_no_local_copy_does_not_log(self):
        content = b'never actually fetched'
        with _isolated_home():
            resource = Resource('test-log-path-noforce', sources=[_FixedBytesSource(content)])
            resource.path(force=False)
            entries = _read_log_entries()
        self.assertEqual([], entries)


if __name__ == '__main__':
    unittest.main()
