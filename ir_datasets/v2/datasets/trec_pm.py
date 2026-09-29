"""TREC Precision Medicine -- a v2 dataset family, covering the two
sub-tasks the 2017/2018 editions ran (MEDLINE abstracts, judged over
``medline-2017-docs``; clinical trials, judged over
``clinicaltrials-2017-docs``) plus 2019, which ran clinical trials only --
all in one file since it's the same track/topic sets throughout, just
scored against different corpora each sub-task, same "track spans multiple
corpora" reasoning as trec_web.py/trec_genomics.py.

topics2017.xml/topics2018.xml each judge both sub-tasks, so
``pm_2017_queries``/``pm_2018_queries`` are shared between the abstracts and
clinical-trials Benchmarks below (only the qrels differ per sub-task); 2019
has no abstracts sub-task, so ``pm_2019_queries`` feeds only the
clinical-trials benchmark.

Docs are imported by reference from ``medline.py`` (``docs_2017``) and
``clinicaltrials.py`` (``docs_2017``/``docs_2019``) -- same cross-file
pattern as ``trec_adhoc.py`` importing ``docs`` from ``disks45.py``.

Previously, the abstracts qrels/benchmarks lived in ``medline.py``
(``medline-trec-pm-2017``/``-2018``) and the clinical-trials qrels/queries/
benchmarks lived in ``clinicaltrials.py``
(``clinicaltrials-trec-pm-2017``/``-2018``/``-2019``) -- moved here and
renamed to drop those corpus prefixes, matching the unprefixed
``trec-web-2002``-style convention every other multi-corpus track file uses.
"""
from ir_datasets.datasets.medline import TrecPm2017Query, TrecPmQuery
from ir_datasets.v2 import Benchmark, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.datasets.clinicaltrials import docs_2017 as clinicaltrials_2017_docs
from ir_datasets.v2.datasets.clinicaltrials import docs_2019 as clinicaltrials_2019_docs
from ir_datasets.v2.datasets.medline import docs_2017 as medline_2017_docs
from ir_datasets.v2.formats import Parser

QREL_DEFS = {
    0: 'not relevant',
    1: 'possibly relevant',
    2: 'definitely relevant',
}


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def __init__(self, qtype, qtype_map=None):
        self.qtype = qtype
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=self.qtype, qtype_map=self.qtype_map, lang=node.lang)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    pm_2017_queries_file = Resource('trec-pm-2017-queries.xml',
        sources=['https://trec.nist.gov/data/precmed/topics2017.xml'],
        md5='16d69bf9119aaaf4b8545c24dde4156d',
        size=5_660,
    )
    pm_2018_queries_file = Resource('trec-pm-2018-queries.xml',
        sources=['http://www.trec-cds.org/topics2018.xml'],
        md5='6c59de8eaf3bd9925a567c50dfab6936',
        size=7_515,
    )
    pm_2019_queries_file = Resource('trec-pm-2019-queries.xml',
        sources=['http://www.trec-cds.org/topics2019.xml'],
        md5='bc26cef87e842837daf2e6680d4652c0',
        size=6_302,
    )

    pm_2017_abstracts_qrels_file = Resource('trec-pm-2017-abstracts-qrels.txt',
        sources=['https://trec.nist.gov/data/precmed/qrels-final-abstracts.txt'],
        md5='0a302cb9cd580709d9e3db9881a25d47',
        size=362_092,
    )
    pm_2018_abstracts_qrels_file = Resource('trec-pm-2018-abstracts-qrels.txt',
        sources=['https://trec.nist.gov/data/precmed/qrels-treceval-abstracts-2018-v2.txt'],
        md5='a09754dec58ee90458ff8e0e7f2cb934',
        size=364_910,
    )
    pm_2017_trials_qrels_file = Resource('trec-pm-2017-trials-qrels.txt',
        sources=['https://trec.nist.gov/data/precmed/qrels-final-trials.txt'],
        md5='3c35f9e62abf64c873250ac8022d5a51',
        size=243_723,
    )
    pm_2018_trials_qrels_file = Resource('trec-pm-2018-trials-qrels.txt',
        sources=['https://trec.nist.gov/data/precmed/qrels-treceval-clinical_trials-2018-v2.txt'],
        md5='a6c9efbecb5f32a19c5ac37f1f98c951',
        size=267_669,
    )
    pm_2019_trials_qrels_file = Resource('trec-pm-2019-trials-qrels.txt',
        sources=['https://trec.nist.gov/data/precmed/qrels-treceval-trials.38.txt'],
        md5='fc4b0cf6007b2dc2a7e81536add65a8c',
        size=243_548,
    )

    # Tables
    # -----------------------------------------
    pm_2017_queries = QueryTable('trec-pm-2017-queries',
        source=pm_2017_queries_file,
        parser=_TrecXmlQueriesParser(TrecPm2017Query),
        count_hint=30,
        citation='dblp:conf/trec/RobertsDVHBLP17',
    )
    pm_2018_queries = QueryTable('trec-pm-2018-queries',
        source=pm_2018_queries_file,
        parser=_TrecXmlQueriesParser(TrecPmQuery),
        count_hint=50,
        citation='dblp:conf/trec/RobertsDVHBL18',
    )
    pm_2019_queries = QueryTable('trec-pm-2019-queries',
        source=pm_2019_queries_file,
        parser=_TrecXmlQueriesParser(TrecPmQuery),
        count_hint=40,
        citation='dblp:conf/trec/RobertsDVHBLPM19',
    )

    pm_2017_abstracts_qrels = TrecQrels('trec-pm-2017-abstracts-qrels',
        source=pm_2017_abstracts_qrels_file, defs=QREL_DEFS, count_hint=22_642)
    pm_2018_abstracts_qrels = TrecQrels('trec-pm-2018-abstracts-qrels',
        source=pm_2018_abstracts_qrels_file, defs=QREL_DEFS, count_hint=22_429)
    pm_2017_trials_qrels = TrecQrels('trec-pm-2017-trials-qrels',
        source=pm_2017_trials_qrels_file, defs=QREL_DEFS, count_hint=13_019)
    pm_2018_trials_qrels = TrecQrels('trec-pm-2018-trials-qrels',
        source=pm_2018_trials_qrels_file, defs=QREL_DEFS, count_hint=14_188)
    pm_2019_trials_qrels = TrecQrels('trec-pm-2019-trials-qrels',
        source=pm_2019_trials_qrels_file, defs=QREL_DEFS, count_hint=12_996)

    # Benchmarks
    # -----------------------------------------
    pm_2017_abstracts = Benchmark('trec-pm-2017-abstracts',
        docs=medline_2017_docs, queries=pm_2017_queries, qrels=pm_2017_abstracts_qrels,
        desc='TREC Precision Medicine 2017, MEDLINE abstracts sub-task.',
        citation='dblp:conf/trec/RobertsDVHBLP17')
    pm_2018_abstracts = Benchmark('trec-pm-2018-abstracts',
        docs=medline_2017_docs, queries=pm_2018_queries, qrels=pm_2018_abstracts_qrels,
        desc='TREC Precision Medicine 2018, MEDLINE abstracts sub-task.',
        citation='dblp:conf/trec/RobertsDVHBL18')
    pm_2017_trials = Benchmark('trec-pm-2017-trials',
        docs=clinicaltrials_2017_docs, queries=pm_2017_queries, qrels=pm_2017_trials_qrels,
        desc='TREC Precision Medicine 2017, clinical trials sub-task.',
        citation='dblp:conf/trec/RobertsDVHBLP17')
    pm_2018_trials = Benchmark('trec-pm-2018-trials',
        docs=clinicaltrials_2017_docs, queries=pm_2018_queries, qrels=pm_2018_trials_qrels,
        desc='TREC Precision Medicine 2018, clinical trials sub-task.',
        citation='dblp:conf/trec/RobertsDVHBL18')
    pm_2019_trials = Benchmark('trec-pm-2019-trials',
        docs=clinicaltrials_2019_docs, queries=pm_2019_queries, qrels=pm_2019_trials_qrels,
        desc='TREC Precision Medicine 2019, clinical trials sub-task.',
        citation='dblp:conf/trec/RobertsDVHBLPM19')


# Registration
# -----------------------------------------
irds.register(pm_2017_abstracts, pm_2018_abstracts, pm_2017_trials, pm_2018_trials, pm_2019_trials)
