import unittest

from ir_datasets.datasets import bright
from ir_datasets.datasets.bright import BrightQrels
from ir_datasets.formats import TrecQrel


class _MockDlc:
    def path(self):
        return 'MOCK'


class TestBrightQrels(unittest.TestCase):
    def test_qrels_include_excluded_ids(self):
        original_parquet_iter = bright.parquet_iter
        bright.parquet_iter = lambda _: iter([{
            'id': 'q1',
            'gold_ids': ['d1', 'd2'],
            'excluded_ids': ['dx', 'N/A', None, 123],
        }])
        try:
            qrels = list(BrightQrels(_MockDlc()).qrels_iter())
        finally:
            bright.parquet_iter = original_parquet_iter

        self.assertEqual([
            TrecQrel('q1', 'd1', 1, '0'),
            TrecQrel('q1', 'd2', 1, '0'),
            TrecQrel('q1', 'dx', -100, '0'),
            TrecQrel('q1', '123', -100, '0'),
        ], qrels)

    def test_qrels_without_excluded_ids(self):
        original_parquet_iter = bright.parquet_iter
        bright.parquet_iter = lambda _: iter([{
            'id': 'q2',
            'gold_ids': ['d3'],
        }])
        try:
            qrels = list(BrightQrels(_MockDlc()).qrels_iter())
        finally:
            bright.parquet_iter = original_parquet_iter

        self.assertEqual([TrecQrel('q2', 'd3', 1, '0')], qrels)
