"""TREC Fair Ranking -- a v2 dataset family.

Self-contained: no import of, or dependency on, another dataset module or
``ir_datasets.registry``/``ir_datasets.load(...)`` reference to one (checked
against the legacy module's imports, all of which are v1-generic
machinery -- ``ir_datasets.util``, ``ir_datasets.formats``,
``ir_datasets.indices`` -- not another dataset).

Two tracks/years, each with its **own** docs corpus (unlike, say,
``trec_adhoc.py``, these do not share a collection across years):

* **2021**: docs (``FairTrecDoc``, Wikipedia articles joined against a
  separate quality/geography metadata file by page id) with two query/qrels
  subsets: ``train`` (one downloaded file serving as *both* queries and
  qrels -- ``FairTrecQueries``/``FairTrecQrels`` each parse the same
  ``rel_docs``-bearing JSON differently) and ``eval`` (separate topics and
  qrels files).
* **2022**: docs (``FairTrec2022Doc``, a wider per-article feature set,
  again joined against a metadata file by id -- via the legacy module's own
  small ``JsonlDocs`` helper, *not* the general-purpose v1/v2 ``JsonlDocs``
  format) with one ``train`` query/qrels subset, again one file serving as
  both (this one plain JSON Lines, not gzipped -- matches v1's own
  ``dlc["2022/train/topics"]``, used with no ``GzipExtract``/``.gunzip()``).

All four custom v1 handlers (``FairTrecDocs``, the legacy module's own
``JsonlDocs``, ``FairTrecQueries``, ``FairTrecQrels``) are reused completely
unmodified via thin ``Parser`` wrappers. Both docs handlers hardcode their
own docstore cache paths internally (``FairTrecDocs.docs_store()`` via the
legacy module's own ``NAME``; the legacy ``JsonlDocs.docs_store()`` via
``self._dlc.path(force=False)``, which -- because v1's ``GzipExtract``
delegates unknown attributes to its wrapped streamer via ``__getattr__`` --
resolves through v2's ``.gunzip()`` pipe straight to the underlying gz
``Resource``'s own path, exactly as it would a v1 ``GzipExtract`` pipeline),
so both are used unmodified, no ``docstore_path=`` override needed -- same
"reuse v1's own path management unmodified" precedent as ``nyt.py``/
``kilt.py``.

Each docs table's ``source=`` is a *two-element list* (the docs file and the
metadata file, both ``.gunzip()``-wrapped ``Resource``s) so
``source_resources`` records real ``derived_from`` edges to both, even
though the parser reads them positionally rather than through some combined
pipeline object.

``docs/trec-fair.yaml`` declares no ``bibtex_ids``; the Benchmarks cite the
TREC 2021/2022 Fair Ranking track overviews (added in the DBLP pass).
"""
from ir_datasets.datasets.trec_fair import (
    FairTrec2022Doc, FairTrec2022TrainQuery, FairTrecDoc,
    FairTrecDocs as _V1FairTrecDocs, FairTrecEvalQuery,
    FairTrecQrels as _V1FairTrecQrels, FairTrecQueries as _V1FairTrecQueries,
    FairTrecQuery, JsonlDocs as _V1JsonlDocs, QREL_DEFS,
)
from ir_datasets.v2 import Benchmark, DocTable, QrelTable, QueryTable, Resource, Source, irds
from ir_datasets.v2.formats import Parser

#: Field mapping for the 2022 corpus's custom ``JsonlDocs`` handler --
#: copied verbatim from legacy ``trec_fair.py``'s ``mapping2022``.
_MAPPING_2022 = {
    'doc_id': 'id', 'title': 'title', 'url': 'url', 'text': 'plain',
    'pred_qual': 'pred_qual', 'qual_cat': 'qual_cat',
    'page_countries': 'page_countries', 'page_subcont_regions': 'page_subcont_regions',
    'source_countries': 'source_countries', 'source_subcont_regions': 'source_subcont_regions',
    'gender': 'gender', 'occupations': 'occupations', 'years': 'years',
    'num_sitelinks': 'num_sitelinks', 'relative_pageviews': 'relative_pageviews',
    'first_letter': 'first_letter', 'creation_date': 'creation_date',
    'first_letter_category': 'first_letter_category', 'gender_category': 'gender_category',
    'creation_date_category': 'creation_date_category', 'years_category': 'years_category',
    'relative_pageviews_category': 'relative_pageviews_category',
    'num_sitelinks_category': 'num_sitelinks_category',
}


class _FairTrecDocsParser(Parser):
    """2021 corpus: article text joined against a separate metadata file by
    page id -- v1's ``FairTrecDocs``, reused unmodified. ``source`` is the
    two-element ``[docs, metadata]`` list from the ``DocTable``'s own
    ``source=`` (see module docstring)."""
    name = 'FairTrecDocs'

    def build(self, source, node):
        docs_source, metadata_source = source
        return _V1FairTrecDocs(docs_source, metadata_source)


class _JsonlDocsParser(Parser):
    """2022 corpus: the legacy module's own small ``JsonlDocs`` helper (not
    the general-purpose v1/v2 ``JsonlDocs`` format), reused unmodified --
    also joins a docs file against a metadata file by id."""
    name = 'JsonlDocs'

    def build(self, source, node):
        docs_source, metadata_source = source
        return _V1JsonlDocs(docs_source, metadata_source, FairTrec2022Doc,
                             _MAPPING_2022, node.count_hint)


class _FairTrecQueriesParser(Parser):
    """Wraps v1's ``FairTrecQueries``, parameterized by which of the three
    query record types (``FairTrecQuery``, ``FairTrecEvalQuery``,
    ``FairTrec2022TrainQuery``) this subset uses."""
    name = 'FairTrecQueries'

    def __init__(self, qtype):
        self.qtype = qtype

    def build(self, source, node):
        return _V1FairTrecQueries(source, self.qtype)


class _FairTrecQrelsParser(Parser):
    """Wraps v1's ``FairTrecQrels`` unmodified -- one qrel per doc id listed
    in a topic's ``rel_docs``, relevance 1."""
    name = 'FairTrecQrels'

    def build(self, source, node):
        return _V1FairTrecQrels(source)


# Files
# -----------------------------------------
docs_2021_file = Resource('trec-fair-2021-docs.json.gz',
    sources=['https://data.boisestate.edu/library/Ekstrand-2021/TRECFairRanking2021/trec_corpus.json.gz'],
    md5='4c1e81d120566a493d5fa90b6114bd49',
    size=15_575_740_862,
)
metadata_2021_file = Resource('trec-fair-2021-metadata.json.gz',
    sources=['https://data.boisestate.edu/library/Ekstrand-2021/TRECFairRanking2021/trec_metadata.json.gz'],
    md5='ae251e9ae0c9fb3a58c3b12e216dcea7',
    size=56_827_296,
)
train_2021_topics_file = Resource('trec-fair-2021-train-topics.json.gz',
    sources=['https://data.boisestate.edu/library/Ekstrand-2021/TRECFairRanking2021/trec_topics.json.gz'],
    md5='bdb72f896833d0c87421b6415d895846',
    size=7_271_598,
)
eval_2021_topics_file = Resource('trec-fair-2021-eval-topics.json.gz',
    sources=['https://drive.google.com/uc?export=download&id=1jGyjB7qOt45jakb32ZtroSkxs5sq5gvU'],
    md5='2e153903c375596914ee9ffdbcefd6a5',
    size=6_055,
)
eval_2021_qrels_file = Resource('trec-fair-2021-eval-qrels.json.gz',
    sources=['https://trec.nist.gov/data/fair/2021-eval-topics-with-qrels.json.gz', Source.mirror()],
    md5='50068634036c00adb54e8be9314bf37c',
    size=120_050,
)
docs_2022_file = Resource('trec-fair-2022-docs.json.gz',
    sources=['https://data.boisestate.edu/library/Ekstrand/TRECFairRanking/corpus/trec_corpus_20220301_plain.json.gz'],
    md5='54661197940765ed5129f0bb0d459a99',
    size=7_677_063_809,
)
metadata_2022_file = Resource('trec-fair-2022-metadata.json.gz',
    sources=['https://data.boisestate.edu/library/Ekstrand/TRECFairRanking/2022/trec_2022_articles_discrete.json.gz'],
    md5='af48525886bae53205f4b64435ae81f2',
    size=236_812_182,
)
train_2022_topics_file = Resource('trec-fair-2022-train-topics.jsonl',
    sources=['https://data.boisestate.edu/library/Ekstrand/TRECFairRanking/2022/trec_2022_train_reldocs.jsonl'],
    md5='d132b4cc8c6c75525479728321db5176',
    size=18_018_410,
)

# Tables
# -----------------------------------------
docs_2021 = DocTable('trec-fair-2021-docs',
    source=[docs_2021_file.gunzip(), metadata_2021_file.gunzip()],
    parser=_FairTrecDocsParser(),
    lang='en',
    count_hint=6_280_328,
)
train_2021_queries = QueryTable('trec-fair-2021-train-queries',
    source=train_2021_topics_file.gunzip(),
    parser=_FairTrecQueriesParser(FairTrecQuery),
    lang='en',
    count_hint=57,
)
train_2021_qrels = QrelTable('trec-fair-2021-train-qrels',
    source=train_2021_topics_file.gunzip(),
    parser=_FairTrecQrelsParser(),
    defs=QREL_DEFS,
    count_hint=2_185_446,
)
eval_2021_queries = QueryTable('trec-fair-2021-eval-queries',
    source=eval_2021_topics_file.gunzip(),
    parser=_FairTrecQueriesParser(FairTrecEvalQuery),
    lang='en',
    count_hint=49,
)
eval_2021_qrels = QrelTable('trec-fair-2021-eval-qrels',
    source=eval_2021_qrels_file.gunzip(),
    parser=_FairTrecQrelsParser(),
    defs=QREL_DEFS,
    count_hint=13_757,
)
docs_2022 = DocTable('trec-fair-2022-docs',
    source=[docs_2022_file.gunzip(), metadata_2022_file.gunzip()],
    parser=_JsonlDocsParser(),
    lang='en',
    count_hint=6_475_537,
)
train_2022_queries = QueryTable('trec-fair-2022-train-queries',
    source=train_2022_topics_file,
    parser=_FairTrecQueriesParser(FairTrec2022TrainQuery),
    lang='en',
    count_hint=50,
)
train_2022_qrels = QrelTable('trec-fair-2022-train-qrels',
    source=train_2022_topics_file,
    parser=_FairTrecQrelsParser(),
    defs=QREL_DEFS,
    count_hint=2_088_306,
)

# Benchmarks
# -----------------------------------------
trec_fair_2021_train = Benchmark('trec-fair-2021-train',
    docs=docs_2021, queries=train_2021_queries, qrels=train_2021_qrels,
    citation='dblp:conf/trec/EkstrandRM021',
    desc='TREC Fair Ranking 2021, official train set.')

trec_fair_2021_eval = Benchmark('trec-fair-2021-eval',
    docs=docs_2021, queries=eval_2021_queries, qrels=eval_2021_qrels,
    citation='dblp:conf/trec/EkstrandRM021',
    desc='TREC Fair Ranking 2021, official evaluation set.')

trec_fair_2022_train = Benchmark('trec-fair-2022-train',
    docs=docs_2022, queries=train_2022_queries, qrels=train_2022_qrels,
    citation='dblp:conf/trec/EkstrandMR022',
    desc='TREC Fair Ranking 2022, official train set.')


# Registration
# -----------------------------------------
irds.register(docs_2021, docs_2022, trec_fair_2021_train, trec_fair_2021_eval, trec_fair_2022_train)
