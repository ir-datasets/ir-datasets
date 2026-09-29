"""NFCorpus -- a v2 dataset family.

Replaces:

    ir_datasets/datasets/nfcorpus.py       wiring + ZipQueries
    ir_datasets/docs/nfcorpus.yaml         desc / bibtex
    ir_datasets/etc/downloads.json         urls / md5 / size   (was central!)
    test/integration/nfcorpus.py           counts / sample records

Counts, content hashes and sample records come from
``python -m ir_datasets.v2.freeze --verify``, not from this file.

One shared docs corpus, nine benchmarks (train/dev/test x main/nontopic/video)
-- all but the "main" three lean on ``Benchmark``'s mix of explicit and
inherited-filtered facets (see the ``*_nontopic``/``*_video`` benchmarks
below): give ``queries=`` explicitly (a subset gets its own, disjoint query
file, not a filtered view of the parent's) while leaving ``qrels`` to fall
through to ``derived_from`` + ``filter`` (the shared parent qrels, filtered
down to that query id set) -- no need to hand-build a ``DerivedTable``, and no
new v2 machinery, just the existing per-facet override in ``Benchmark.edge``.

The "main" query sets need something not yet needed anywhere else in v2,
though: each one is *two* parallel single-column TSVs -- titles, and a second
field ("all" text, or a video description) -- meant to become one query
record with both columns (v1's ``ZipQueries``). That is genuinely a "combine
two tables column-wise" operation, which ``Filter``/``DerivedTable`` don't do
(they filter rows, not merge columns) -- so it gets a small, dataset-local
``_ZipQueriesParser`` below rather than a new core primitive. Nothing else in
v2 needs this yet, and the two record shapes here (title+all vs.
title+description) are different enough that a generic version would need its
own design pass.
"""
from typing import NamedTuple

from ir_datasets import formats as _v1
from ir_datasets.formats import GenericQuery

import ir_datasets
from ir_datasets.v2 import (
    Benchmark, Filter, Parser, QueryTable, Resource, Source, TrecQrels, TsvDocs,
    TsvQueries, ids_from_lines, irds,
)

BASE = ir_datasets.util.home_path() / 'nfcorpus'

CITATION = 'dblp:conf/ecir/BotevaGSR16'

QREL_DEFS = {
    2: "A direct link from the query to the document the cited sources section of a page.",
    1: "A link exists from the query to another query that directly links to the document.",
    0: "Marginally relevant, based on topic containment.",
}


class NfCorpusDoc(NamedTuple):
    doc_id: str
    url: str
    title: str
    abstract: str
    def default_text(self):
        return f'{self.title} {self.abstract}'


class NfCorpusQuery(NamedTuple):
    query_id: str
    title: str
    all: str
    def default_text(self):
        return self.title


class NfCorpusVideoQuery(NamedTuple):
    query_id: str
    title: str
    desc: str
    def default_text(self):
        return self.title


class _ZipQueriesParser(Parser):
    """Two parallel single-column query TSVs (title file + a second field),
    zipped positionally into one composite record -- v1's ``ZipQueries``, as
    a dataset-local ``Parser`` (see the module docstring for why this isn't a
    core v2 primitive). Each underlying file is read through v1's own
    ``TsvQueries`` (a ``GenericQuery`` per row); this only re-packs the two
    parallel iterators together, assuming (as v1 did) that both files list
    the same query ids in the same order.
    """
    name = 'ZipQueries'

    def __init__(self, cls):
        self.cls = cls

    def _iter(self, sources):
        first, second = sources
        first_iter = _v1.TsvQueries(first, query_cls=GenericQuery).queries_iter()
        second_iter = _v1.TsvQueries(second, query_cls=GenericQuery).queries_iter()
        for q1, q2 in zip(first_iter, second_iter):
            assert q1.query_id == q2.query_id, \
                f'misaligned zip: {q1.query_id!r} != {q2.query_id!r}'
            yield self.cls(q1.query_id, q1.text, q2.text)

    def build(self, source, node):
        handler = type('_ZipQueriesHandler', (), {})()
        handler.queries_iter = lambda: self._iter(source)
        handler.queries_cls = lambda: self.cls
        return handler


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    main_file = Resource('nfcorpus-main.tar.gz',
        sources=['https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz',
                 Source.mirror()],
        hash='md5:49c061fbadc52ba4d35d0e42e2d742fd',
        size=31_039_523,
    )
    nontopic_ids_file = main_file.member('nfcorpus/nontopics.ids').cache(BASE / 'nontopics.ids')
    video_ids_file = main_file.member('nfcorpus/all_videos.ids').cache(BASE / 'all_videos.ids')

    # Tables
    # -----------------------------------------
    docs = TsvDocs('nfcorpus-docs',
        source=main_file.member('nfcorpus/raw/doc_dump.txt').cache(BASE / 'collection.tsv'),
        cls=NfCorpusDoc,
        count_hint=9_964,
    )

    # Per-split tables + benchmarks
    # -----------------------------------------
    # One block per split (train/dev/test): the shared docs corpus plus that
    # split's own queries/qrels, and its nontopic/video variants --
    # structurally identical across the three, so it is a loop rather than
    # copy-pasted thrice.
    _benchmarks = {}
    for _split in ('train', 'dev', 'test'):
        title_member = main_file.member(f'nfcorpus/{_split}.titles.queries').cache(
            BASE / _split / 'queries.titles.tsv')
        all_member = main_file.member(f'nfcorpus/{_split}.all.queries').cache(
            BASE / _split / 'queries.all.tsv')
        queries = QueryTable(f'nfcorpus-{_split}-queries',
            source=[title_member, all_member],
            parser=_ZipQueriesParser(NfCorpusQuery),
        )
        qrels = TrecQrels(f'nfcorpus-{_split}-qrels',
            source=main_file.member(f'nfcorpus/{_split}.3-2-1.qrel').cache(
                BASE / _split / 'qrels'),
            defs=QREL_DEFS,
        )
        main_bm = Benchmark(f'nfcorpus-{_split}',
            docs=docs, queries=queries, qrels=qrels,
            citation=CITATION,
            desc=f'Official {_split} set of NFCorpus. Queries include both a '
                 '"title" and a combined "all" text field (titles, '
                 'descriptions, topics, transcripts and comments).')

        nontopic_queries = TsvQueries(f'nfcorpus-{_split}-nontopic-queries',
            source=main_file.member(f'nfcorpus/{_split}.nontopic-titles.queries').cache(
                BASE / _split / 'nontopic' / 'queries.tsv'),
        )
        nontopic_bm = Benchmark(f'nfcorpus-{_split}-nontopic',
            queries=nontopic_queries,
            derived_from=main_bm,
            filter=Filter(query_ids=ids_from_lines(nontopic_ids_file), mode='include'),
            citation=CITATION,
            desc=f'Official {_split} set, filtered to exclude queries from topic pages.')

        vid_title_member = main_file.member(f'nfcorpus/{_split}.vid-titles.queries').cache(
            BASE / _split / 'video' / 'queries.titles.tsv')
        vid_desc_member = main_file.member(f'nfcorpus/{_split}.vid-desc.queries').cache(
            BASE / _split / 'video' / 'queries.desc.tsv')
        video_queries = QueryTable(f'nfcorpus-{_split}-video-queries',
            source=[vid_title_member, vid_desc_member],
            parser=_ZipQueriesParser(NfCorpusVideoQuery),
        )
        video_bm = Benchmark(f'nfcorpus-{_split}-video',
            queries=video_queries,
            derived_from=main_bm,
            filter=Filter(query_ids=ids_from_lines(video_ids_file), mode='include'),
            citation=CITATION,
            desc=f'Official {_split} set, filtered to only include queries from video pages.')

        _benchmarks[_split] = main_bm
        _benchmarks[f'{_split}-nontopic'] = nontopic_bm
        _benchmarks[f'{_split}-video'] = video_bm


# Registration
# -----------------------------------------
irds.register(*_benchmarks.values())
