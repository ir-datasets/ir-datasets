"""TREC Microblog -- a v2 dataset family, covering the two years judged over
tweets2013-ia (2013/2014), same "track spans multiple corpora" reasoning as
trec_web.py/trec_genomics.py -- even though, for now, tweets2013-ia is the
only corpus this track judges (see ``tweets2013_ia.py``'s docstring; a
track's home shouldn't depend on how many corpora happen to be migrated for
it yet).

Docs are imported by reference from ``tweets2013_ia.py`` -- same cross-file
pattern as ``trec_adhoc.py`` importing ``docs`` from ``disks45.py``.

Previously bundled in ``tweets2013_ia.py`` as
``tweets2013-ia-trec-mb-2013/2014`` -- moved here and renamed to drop the
corpus prefix, matching the unprefixed ``trec-web-2002``-style convention
every other multi-corpus track file uses.
"""
from ir_datasets.datasets.tweets2013_ia import QREL_DEFS, QTYPE_MAP_13, QTYPE_MAP_14, RM_TAGS, TrecMb13Query, TrecMb14Query
from ir_datasets.v2 import Benchmark, Resource, Source, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.tweets2013_ia import docs as tweets2013_ia_docs

CITATION_2013 = 'dblp:conf/trec/LinE13'
CITATION_2014 = 'dblp:conf/trec/LinWES14'

# Files
# -----------------------------------------
queries_2013_file = Resource('trec-mb-2013-queries.txt',
    sources=['https://trec.nist.gov/data/microblog/2013/topics.MB111-170.txt', Source.irds()],
    md5='0b78d99dfa2d655dca7e9f138a93c21a',
    size=11_471,
)
qrels_2013_file = Resource('trec-mb-2013-qrels.txt',
    sources=['https://trec.nist.gov/data/microblog/2013/qrels.txt', Source.irds()],
    md5='4776a5dfd80b3f675184315ec989c02f',
    size=1_995_812,
)
queries_2014_file = Resource('trec-mb-2014-queries.txt',
    sources=['https://trec.nist.gov/data/microblog/2014/topics.desc.MB171-225.txt', Source.irds()],
    md5='e9d520f976176e710fd68bb3a065a3e7',
    size=17_785,
)
qrels_2014_file = Resource('trec-mb-2014-qrels.txt',
    sources=['https://trec.nist.gov/data/microblog/2014/qrels2014.txt', Source.irds()],
    md5='68d9a1920b244f6ccdc687ee1d473214',
    size=1_623_580,
)

# Tables
# -----------------------------------------
queries_2013 = TrecQueries('trec-mb-2013-queries',
    source=queries_2013_file,
    qtype=TrecMb13Query, qtype_map=QTYPE_MAP_13, remove_tags=RM_TAGS,
    lang='en', count_hint=60, citation=CITATION_2013,
)
qrels_2013 = TrecQrels('trec-mb-2013-qrels',
    source=qrels_2013_file, defs=QREL_DEFS, count_hint=71_279)

queries_2014 = TrecQueries('trec-mb-2014-queries',
    source=queries_2014_file,
    qtype=TrecMb14Query, qtype_map=QTYPE_MAP_14, remove_tags=RM_TAGS,
    lang='en', count_hint=55, citation=CITATION_2014,
)
qrels_2014 = TrecQrels('trec-mb-2014-qrels',
    source=qrels_2014_file, defs=QREL_DEFS, count_hint=57_985)

# Benchmarks
# -----------------------------------------
mb_2013 = Benchmark('trec-mb-2013',
    docs=tweets2013_ia_docs, queries=queries_2013, qrels=qrels_2013,
    desc='TREC Microblog 2013.')
mb_2014 = Benchmark('trec-mb-2014',
    docs=tweets2013_ia_docs, queries=queries_2014, qrels=qrels_2014,
    desc='TREC Microblog 2014.')


# Registration
# -----------------------------------------
irds.register(mb_2013, mb_2014)
