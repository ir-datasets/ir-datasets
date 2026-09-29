"""TREC News Track (background linking task) -- a v2 dataset family,
covering the three years it ran: 2018/2019/2020, all judged against WaPo
(2018/2019 over ``wapo-v2-docs``; 2020 has no docs facet at all -- v1 itself
registers that subset queries+qrels only, no ``v3`` collection ever defined,
preserved as-is here).

Docs are imported by reference from ``wapo.py`` -- same cross-file pattern
as ``trec_adhoc.py`` importing ``docs`` from ``disks45.py``.

Previously bundled in ``wapo.py`` as ``wapo-trec-news-2018/2019/2020``
(alongside ``wapo-trec-core-2018``, now split out separately into
``trec_core.py``) -- moved here and renamed to drop the corpus prefix,
matching the unprefixed ``trec-web-2002``-style convention every other
multi-corpus track file uses. See ``trec_core.py``'s module docstring for why
this overturns the old single-consumer bundling convention.
"""
from ir_datasets.datasets.wapo import BL_MAP, BL_QREL_DEFS, RM_TAGS, TrecBackgroundLinkingQuery
from ir_datasets.v2 import Benchmark, Resource, Source, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.wapo import docs_v2 as wapo_v2_docs

CITATION_2018 = 'dblp:conf/trec/SoboroffHH18'
CITATION_2019 = 'Soboroff2019News'

with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    news_2018_queries_file = Resource('trec-news-2018-queries.txt',
        sources=['https://trec.nist.gov/data/news/2018/newsir18-topics.txt', Source.irds()],
        md5='73740793543b439d1ff1b8ee9359973a',
        size=12_489,
    )
    news_2018_qrels_file = Resource('trec-news-2018-qrels.txt',
        sources=['https://trec.nist.gov/data/news/2018/bqrels.exp-gains.txt', Source.irds()],
        md5='396963175006cb3201ea7c16e874033a',
        size=364_062,
    )
    news_2019_queries_file = Resource('trec-news-2019-queries.xml',
        sources=['https://trec.nist.gov/data/news/2019/newsir19-background-linking-topics.xml', Source.irds()],
        md5='388b5c96f8962da17eb1024b856d21c1',
        size=14_847,
    )
    news_2019_qrels_file = Resource('trec-news-2019-qrels.txt',
        sources=['https://trec.nist.gov/data/news/2019/newsir19-qrels-background.txt', Source.irds()],
        md5='7b839a1a94e349d3facf28012542cc1d',
        size=669_632,
    )
    news_2020_queries_file = Resource('trec-news-2020-queries.txt',
        sources=['https://trec.nist.gov/data/news/2020/newsir20-topics.txt', Source.irds()],
        md5='2674538a07fb7ac29200cbc4c4a05404',
        size=13_217,
    )
    news_2020_qrels_file = Resource('trec-news-2020-qrels.txt',
        sources=['https://trec.nist.gov/data/news/2020/qrels.background', Source.irds()],
        md5='7c31f731775bdd4148d349df1a9e43fc',
        size=729_348,
    )

    # Tables
    # -----------------------------------------
    news_2018_queries = TrecQueries('trec-news-2018-queries',
        source=news_2018_queries_file,
        qtype=TrecBackgroundLinkingQuery, qtype_map=BL_MAP, remove_tags=RM_TAGS,
        count_hint=50)
    news_2018_qrels = TrecQrels('trec-news-2018-qrels',
        source=news_2018_qrels_file, defs=BL_QREL_DEFS, count_hint=8_508)

    news_2019_queries = TrecQueries('trec-news-2019-queries',
        source=news_2019_queries_file,
        qtype=TrecBackgroundLinkingQuery, qtype_map=BL_MAP, remove_tags=RM_TAGS,
        count_hint=60)
    news_2019_qrels = TrecQrels('trec-news-2019-qrels',
        source=news_2019_qrels_file, defs=BL_QREL_DEFS, count_hint=15_655)

    news_2020_queries = TrecQueries('trec-news-2020-queries',
        source=news_2020_queries_file,
        qtype=TrecBackgroundLinkingQuery, qtype_map=BL_MAP, remove_tags=RM_TAGS,
        count_hint=50)
    news_2020_qrels = TrecQrels('trec-news-2020-qrels',
        source=news_2020_qrels_file, defs=BL_QREL_DEFS, count_hint=17_764)

    # Benchmarks
    # -----------------------------------------
    news_2018 = Benchmark('trec-news-2018',
        docs=wapo_v2_docs, queries=news_2018_queries, qrels=news_2018_qrels,
        citation=CITATION_2018,
        desc='TREC News Track 2018 background linking task.')
    news_2019 = Benchmark('trec-news-2019',
        docs=wapo_v2_docs, queries=news_2019_queries, qrels=news_2019_qrels,
        citation=CITATION_2019,
        desc='TREC News Track 2019 background linking task.')
    news_2020 = Benchmark('trec-news-2020',
        queries=news_2020_queries, qrels=news_2020_qrels,
        desc='TREC News Track 2020 background linking task. v1 registers this '
             'with no docs facet (no v3 collection was ever defined) -- preserved as-is.')


# Registration
# -----------------------------------------
irds.register(news_2018, news_2019, news_2020)
