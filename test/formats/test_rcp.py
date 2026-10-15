import os
import tempfile
import unittest
from ir_datasets.v2.formats import RcpQrel, RcpQrels, _RcpQrelsParser


class _Source:
    def __init__(self, path):
        self._path = path

    def path(self):
        return self._path


class TestRcpQrels(unittest.TestCase):

    def test_core(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'qrels.parquet')
            pq.write_table(pa.table({
                'query-id': ['1', '1'], 'corpus-id': ['a', 'b'],
                'score': [1, 1], 'gain': [0.9, 0.0], 'theta': [2.5, None],
            }), path)
            handler = _RcpQrelsParser().build(_Source(path), None)
            self.assertEqual(list(handler.qrels_iter()), [
                RcpQrel('1', 'a', 0.9, 2.5),
                RcpQrel('1', 'b', 0.0, None),  # human-positive, outside the judged pool
            ])
            self.assertIs(handler.qrels_cls(), RcpQrel)
            self.assertEqual(handler.qrels_defs(), {})

    def test_no_score_tally(self):
        self.assertIsNone(RcpQrels.score_field)


if __name__ == '__main__':
    unittest.main()
