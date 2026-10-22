"""MS MARCO -- the family as one bundle.

The five corpora (passage and document, each in its v1 and v2 form, plus QnA) and every
benchmark defined over them are separate nodes in their own modules; this registers an
``Overview`` that holds them together; its documentation page says what they are and how they
relate. It adds no data of its own. (TREC Deep Learning judges these corpora against its own
query sets, so it stays in ``trec_dl.py``; the page points to it.)
"""
from ir_datasets.v2 import Benchmark, DocTable, Overview, irds
# The members register themselves on import; the bundle is built from what they registered.
from ir_datasets.v2.datasets import (  # noqa: F401
    msmarco_document, msmarco_document_v2, msmarco_passage, msmarco_passage_v2, msmarco_qna,
)
from ir_datasets.v2.datasets.msmarco_passage import CITATION


_members = [name for name, node in irds.nodes.items()
            if name.startswith('irds:msmarco') and isinstance(node, (DocTable, Benchmark))]

irds.register(Overview(
    'msmarco', parts=_members, citation=CITATION))
