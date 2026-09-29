"""CLEF eHealth -- a v2 dataset family (the CLEF eHealth IR task, English
plus six language variants of the same query set, all judging
``clueweb12-b13-docs``, imported by reference from ``clueweb12.py`` -- same
cross-file pattern as ``disks45.py``/``trec_adhoc.py``).

CLEF eHealth is not a TREC track, so -- like ``ntcir_www.py`` -- it gets its
own file rather than living in ``clueweb12.py`` or ``trec_misinfo.py``.

All 7 language variants (English, ``cs``/``de``/``fr``/``hu``/``pl``/``sv``)
share the *same* six underlying qrels/qtrust/qunder(-shaped) files -- 2016 and
2017 -- via v1's ``EhealthQrels``, reused completely unmodified via a local
``Parser``. ``EhealthQrels`` takes three parallel lists of dlcs (``qrels``,
``qtrust``, ``qunder``, each ``[2016_file, 2017_file]``) plus a
``query_id_suffix`` (empty for English, ``'-xx'`` for the others) -- this is
how the *same* qrels rows are disambiguated per language: ``qrels_iter``
appends the suffix to every query_id it yields, rather than the underlying
files differing at all. So all 7 ``QrelTable``s below share one module-level
``_QRELS_SOURCE`` (the same six Resources, nested as
``([qrels2016, qrels2017], [qtrust2016, qtrust2017], [qunder2016, qunder2017])``
-- ``source_resources()`` walks nested lists/tuples of any depth, so this
still produces correct ``derived_from`` edges), differing only in the
``query_id_suffix`` baked into each one's ``Parser`` instance.

One extra wrinkle preserved exactly as v1 has it (not a typo): the "qunder"
role for 2017 is actually played by a file named ``qreads`` upstream
(``dlc['clef-ehealth/2017.qreads']``) -- read v1's own ``_init()`` and this
is genuinely what it passes as the second element of the ``qunder_dlcs``
list, so ``qunder_2017_file`` below is that file, used the same way.

Each of the 7 query sets is its own small XML download, wrapped in v1's
``FixAmp`` (a stream wrapper that fixes unescaped ``&`` in the source XML --
v1 applies this only to queries, never to qrels) at *parse* time, inside the
``Parser.build()`` below -- not as part of each Resource's ``source=``, so
that ``source_resources()`` still walks down to the real Resource for
``derived_from`` edges (``FixAmp`` has no ``._parent`` a Readable-walking
edge-collector could follow, so it must stay a build()-time wrapper, exactly
like ``ClueWeb12b13Extractor`` in ``clueweb12.py`` stays a build()-time
wrapper rather than part of ``source=``). ``GenericQuery``/``ehealth_map``
(``{'id': 'query_id', 'title': 'text'}``) are used for all 7 (v1 never
defined a dedicated eHealth query record type). ``EHEALTH_QREL_DEFS`` and
``ehealth_map`` are still imported from legacy ``ir_datasets.datasets.
clueweb12`` -- constant imports, not behavioral ones.

Per language: 300 queries, 269,232 qrels each (identical qrels content
across all 7, differing only in query_id suffix -- see
``ir_datasets/etc/metadata.json``).
"""
from ir_datasets.datasets.clueweb12 import EHEALTH_QREL_DEFS, FixAmp, ehealth_map
from ir_datasets.datasets.clueweb12 import EhealthQrels as _V1EhealthQrels
from ir_datasets.formats import GenericQuery
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.datasets.clueweb12 import docs_b13
from ir_datasets.v2.formats import Parser

CITATION = 'dblp:conf/clef/ZucconPGKLPMBD16; dblp:conf/clef/PalottiZJPLGKH17'


class _EhealthQueriesParser(Parser):
    """Wraps the raw XML source in v1's ``FixAmp`` at build time -- see
    module docstring for why this can't be part of ``source=`` directly."""
    name = 'TrecXmlQueries(FixAmp)'

    def __init__(self, qtype_map):
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(FixAmp(source), qtype=GenericQuery, qtype_map=self.qtype_map, lang=node.lang)


class _EhealthQrelsParser(Parser):
    name = 'EhealthQrels'

    def __init__(self, query_id_suffix=''):
        self.query_id_suffix = query_id_suffix

    def build(self, source, node):
        qrels_dlcs, qtrust_dlcs, qunder_dlcs = source
        return _V1EhealthQrels(qrels_dlcs, qtrust_dlcs, qunder_dlcs, node.defs or {},
                                query_id_suffix=self.query_id_suffix)


# Files
# -----------------------------------------
qrels_2016_file = Resource('clef-ehealth-2016.qrels',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2016Task3/master/qrels/task1.qrels'],
    md5='5392a6f7cdbb0cab56c34656ab100684',
    size=5_550_000,
)
qrels_2017_file = Resource('clef-ehealth-2017.qrels',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/assessments/2017/clef2017_qrels.txt'],
    md5='b9909f2fa7f2a0ceca1033fc92729482',
    size=4_411_584,
)
qtrust_2016_file = Resource('clef-ehealth-2016.qtrust',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2016Task3/master/qrels/task1.qtrust'],
    md5='14cf266961f686b49d8430b802064ac6',
    size=5_669_328,
)
qtrust_2017_file = Resource('clef-ehealth-2017.qtrust',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/assessments/2017/clef2017_qtrust.txt'],
    md5='3a43cc9a49a781b13f0b8438bceed2f5',
    size=4_528_752,
)
# v1's "qunder" role for 2016 is a file actually named qunder upstream...
qunder_2016_file = Resource('clef-ehealth-2016.qunder',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2016Task3/master/qrels/task1.qunder'],
    md5='4314d1a0db76a50204e5e900ffa2d4e3',
    size=5_695_332,
)
# ...but for 2017 it's a file actually named qreads upstream -- not a typo,
# see module docstring; it plays the exact same "qunder" role in EhealthQrels.
qunder_2017_file = Resource('clef-ehealth-2017.qreads',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/assessments/2017/clef2017_qreads.txt'],
    md5='75302c012adf0126ab93e330ffb6aaab',
    size=4_531_566,
)

_QRELS_SOURCE = (
    [qrels_2016_file, qrels_2017_file],
    [qtrust_2016_file, qtrust_2017_file],
    [qunder_2016_file, qunder_2017_file],
)

queries_en_file = Resource('clef-ehealth-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2016Task3/master/eng_queries/queries2016_with_url.xml'],
    md5='ed0289056e21643baa07dfc7bee9574b',
    size=47_427,
)
queries_cs_file = Resource('clef-ehealth-cs-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/queries/multilingual/queries2016cs.xml'],
    md5='f7940a8961fae713742b935c83b6118d',
    size=33_079,
)
queries_de_file = Resource('clef-ehealth-de-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/queries/multilingual/queries2016de.xml'],
    md5='f59e000328a8cbe357ff94d5d5e82474',
    size=31_575,
)
queries_fr_file = Resource('clef-ehealth-fr-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/queries/multilingual/queries2016fr.xml'],
    md5='7362ccb451a606af7a47e359894f524c',
    size=35_744,
)
queries_hu_file = Resource('clef-ehealth-hu-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/queries/multilingual/queries2016hu.xml'],
    md5='17603aecae72eeb70ada73700956ab3f',
    size=31_670,
)
queries_pl_file = Resource('clef-ehealth-pl-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/queries/multilingual/queries2016pl.xml'],
    md5='e6d442846786793c235bf927289af4c3',
    size=32_741,
)
queries_sv_file = Resource('clef-ehealth-sv-queries.xml',
    sources=['https://raw.githubusercontent.com/CLEFeHealth/CLEFeHealth2017IRtask/master/queries/multilingual/queries2016sv.xml'],
    md5='bc9066f128a391c9f5c282dbcca3e44f',
    size=30_669,
)

# Tables
# -----------------------------------------
queries_en = QueryTable('clef-ehealth-queries',
    source=queries_en_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='en', count_hint=300, citation=CITATION)
queries_cs = QueryTable('clef-ehealth-cs-queries',
    source=queries_cs_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='cs', count_hint=300, citation=CITATION)
queries_de = QueryTable('clef-ehealth-de-queries',
    source=queries_de_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='de', count_hint=300, citation=CITATION)
queries_fr = QueryTable('clef-ehealth-fr-queries',
    source=queries_fr_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='fr', count_hint=300, citation=CITATION)
queries_hu = QueryTable('clef-ehealth-hu-queries',
    source=queries_hu_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='hu', count_hint=300, citation=CITATION)
queries_pl = QueryTable('clef-ehealth-pl-queries',
    source=queries_pl_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='pl', count_hint=300, citation=CITATION)
queries_sv = QueryTable('clef-ehealth-sv-queries',
    source=queries_sv_file, parser=_EhealthQueriesParser(ehealth_map),
    lang='sv', count_hint=300, citation=CITATION)

qrels_en = QrelTable('clef-ehealth-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix=''),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)
qrels_cs = QrelTable('clef-ehealth-cs-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix='-cs'),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)
qrels_de = QrelTable('clef-ehealth-de-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix='-de'),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)
qrels_fr = QrelTable('clef-ehealth-fr-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix='-fr'),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)
qrels_hu = QrelTable('clef-ehealth-hu-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix='-hu'),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)
qrels_pl = QrelTable('clef-ehealth-pl-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix='-pl'),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)
qrels_sv = QrelTable('clef-ehealth-sv-qrels',
    source=_QRELS_SOURCE, parser=_EhealthQrelsParser(query_id_suffix='-sv'),
    defs=EHEALTH_QREL_DEFS, count_hint=269_232, citation=CITATION)

# Benchmarks
# -----------------------------------------
clef_ehealth = Benchmark('clef-ehealth',
    docs=docs_b13, queries=queries_en, qrels=qrels_en,
    citation=CITATION, desc='CLEF eHealth IR task (English).')
clef_ehealth_cs = Benchmark('clef-ehealth-cs',
    docs=docs_b13, queries=queries_cs, qrels=qrels_cs,
    citation=CITATION, desc='CLEF eHealth IR task (Czech queries).')
clef_ehealth_de = Benchmark('clef-ehealth-de',
    docs=docs_b13, queries=queries_de, qrels=qrels_de,
    citation=CITATION, desc='CLEF eHealth IR task (German queries).')
clef_ehealth_fr = Benchmark('clef-ehealth-fr',
    docs=docs_b13, queries=queries_fr, qrels=qrels_fr,
    citation=CITATION, desc='CLEF eHealth IR task (French queries).')
clef_ehealth_hu = Benchmark('clef-ehealth-hu',
    docs=docs_b13, queries=queries_hu, qrels=qrels_hu,
    citation=CITATION, desc='CLEF eHealth IR task (Hungarian queries).')
clef_ehealth_pl = Benchmark('clef-ehealth-pl',
    docs=docs_b13, queries=queries_pl, qrels=qrels_pl,
    citation=CITATION, desc='CLEF eHealth IR task (Polish queries).')
clef_ehealth_sv = Benchmark('clef-ehealth-sv',
    docs=docs_b13, queries=queries_sv, qrels=qrels_sv,
    citation=CITATION, desc='CLEF eHealth IR task (Swedish queries).')


# Registration
# -----------------------------------------
irds.register(clef_ehealth, clef_ehealth_cs, clef_ehealth_de, clef_ehealth_fr,
              clef_ehealth_hu, clef_ehealth_pl, clef_ehealth_sv)
