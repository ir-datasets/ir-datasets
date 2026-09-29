"""ClueWeb09 -- a v2 dataset family (the CMU/Lemur ClueWeb 2009 web crawl,
~1B pages across 10 languages).

The corpus is LDC/NIST-gated and distributed on hard drives, not
automatically downloadable -- same shape as ``disks45.py``/``trec_arabic.py``:
a ``Directory`` Resource pointed at by ``Source.external()``, at
``<home>/external/clueweb09`` (old v1 location ``<home>/clueweb09/corpus``
still read as a fallback).
A separately downloadable "chk" archive (per-WARC-file record-offset
checkpoints, used for random access / seeking within ``docs.lookup``) is a
real downloadable ``Resource``, extracted via v1's ``TarExtractAll`` the same
way ``c4.py`` extracts its checkpoints archive.

Reuses v1's ``ClueWeb09Docs`` (a ``WarcDocs`` subclass) directly as a v2
``parser=`` wrapper -- ``WarcDocs`` has no general-purpose v2 format node, so
it goes through a local ``Parser``, same "thin layer over v1 machinery"
pattern as every other migrated family.

v1 registers 11 language-filtered views (``ar``/``zh``/``en``/``fr``/``de``/
``it``/``ja``/``ko``/``pt``/``es``) plus ``catb`` (the ``ClueWeb09_English_1``
subset used by the TREC Web Track's Category B runs) over the *same*
underlying directory tree -- each is its own ``DocTable`` here (``source=``
is the same two Resources for all of them; the ``dirs=`` filter is baked
into the parser instance, not the source, so each still records the same
``derived_from`` edges).

TREC Web Track 2009-2012 (ad hoc + diversity, over both the full English and
catb subsets) and the TREC 2009 Million Query track that judge this corpus
are *not* defined here -- see ``trec_web.py`` and ``trec_mq.py``, which
import ``docs_en``/``docs_catb``/``docs`` from this file by reference, same
cross-file pattern as ``disks45.py``/``trec_adhoc.py``.
"""
import ir_datasets
from ir_datasets.datasets.clueweb09 import ClueWeb09Docs as _V1ClueWeb09Docs
from ir_datasets.util import TarExtractAll
from ir_datasets.v2 import Directory, DocTable, Resource, Source, irds
from ir_datasets.v2.formats import Parser

DUA = ("ClueWeb09 is distributed on hard drives by CMU "
       "<https://lemurproject.org/clueweb09/>; see "
       "<https://ir-datasets.com/ClueWeb09.html#DataAccess> for the procedure.")

DATA_ACCESS = (
    "ClueWeb09 is available on hard drives from CMU <https://lemurproject.org/clueweb09/>. "
    "More details about the procedure can be found at "
    "<https://ir-datasets.com/ClueWeb09.html#DataAccess>. "
    "Link the ClueWeb09 source files here: {path}\n"
    "Should contain directories like ClueWeb09_English_1"
)

DOCS_DEFAULT_PATH = 'clueweb09'
DOCS_OLD_LOCATION = 'clueweb09/corpus'
BASE_PATH = ir_datasets.util.home_path() / 'clueweb09'

_ALL_DIRS = ['ClueWeb09_Arabic_1', 'ClueWeb09_Chinese_1', 'ClueWeb09_Chinese_2', 'ClueWeb09_Chinese_3',
             'ClueWeb09_Chinese_4', 'ClueWeb09_English_1', 'ClueWeb09_English_2', 'ClueWeb09_English_3',
             'ClueWeb09_English_4', 'ClueWeb09_English_5', 'ClueWeb09_English_6', 'ClueWeb09_English_7',
             'ClueWeb09_English_8', 'ClueWeb09_English_9', 'ClueWeb09_English_10', 'ClueWeb09_French_1',
             'ClueWeb09_German_1', 'ClueWeb09_Italian_1', 'ClueWeb09_Japanese_1', 'ClueWeb09_Japanese_2',
             'ClueWeb09_Korean_1', 'ClueWeb09_Portuguese_1', 'ClueWeb09_Spanish_1', 'ClueWeb09_Spanish_2']


class _ClueWeb09DocsParser(Parser):
    name = 'ClueWeb09Docs'

    def __init__(self, dirs=None):
        self.dirs = dirs

    def build(self, source, node):
        docs_source, chk_source = source
        chk = TarExtractAll(chk_source, BASE_PATH / 'corpus.chk')
        return _V1ClueWeb09Docs(docs_source, chk, dirs=self.dirs, lang=node.lang)


# Files
# -----------------------------------------
docs_file = Directory('clueweb09.dir',
    sources=[Source.external(DOCS_DEFAULT_PATH, old_locations=[DOCS_OLD_LOCATION], instructions=DATA_ACCESS)],
    dua=DUA,
)
chk_file = Resource('clueweb09-chk.tar.gz',
    sources=['https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/ir-datasets/clueweb09/clueweb09-source-chk.tar.gz'],
    hash='md5:74328d9c743c52ddef434ce41a4e6dc1',
    size=3_582_668_561,
)

# Tables
# -----------------------------------------
docs = DocTable('clueweb09',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(),
    count_hint=1_040_859_705)
docs_ar = DocTable('clueweb09-ar',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Arabic_1']),
    lang='ar', count_hint=29_192_662)
docs_zh = DocTable('clueweb09-zh',
    source=[docs_file, chk_file],
    parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Chinese_1', 'ClueWeb09_Chinese_2',
                                       'ClueWeb09_Chinese_3', 'ClueWeb09_Chinese_4']),
    lang='zh', count_hint=177_489_357)
docs_en = DocTable('clueweb09-en',
    source=[docs_file, chk_file],
    parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_English_1', 'ClueWeb09_English_2', 'ClueWeb09_English_3',
                                       'ClueWeb09_English_4', 'ClueWeb09_English_5', 'ClueWeb09_English_6',
                                       'ClueWeb09_English_7', 'ClueWeb09_English_8', 'ClueWeb09_English_9',
                                       'ClueWeb09_English_10']),
    lang='en', count_hint=503_903_810)
docs_fr = DocTable('clueweb09-fr',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_French_1']),
    lang='fr', count_hint=50_883_172)
docs_de = DocTable('clueweb09-de',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_German_1']),
    lang='de', count_hint=49_814_309)
docs_it = DocTable('clueweb09-it',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Italian_1']),
    lang='it', count_hint=27_250_729)
docs_ja = DocTable('clueweb09-ja',
    source=[docs_file, chk_file],
    parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Japanese_1', 'ClueWeb09_Japanese_2']),
    lang='ja', count_hint=67_337_717)
docs_ko = DocTable('clueweb09-ko',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Korean_1']),
    lang='ko', count_hint=18_075_141)
docs_pt = DocTable('clueweb09-pt',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Portuguese_1']),
    lang='pt', count_hint=37_578_858)
docs_es = DocTable('clueweb09-es',
    source=[docs_file, chk_file],
    parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_Spanish_1', 'ClueWeb09_Spanish_2']),
    lang='es', count_hint=79_333_950)
docs_catb = DocTable('clueweb09-catb',
    source=[docs_file, chk_file], parser=_ClueWeb09DocsParser(dirs=['ClueWeb09_English_1']),
    lang='en', count_hint=50_220_423)


# Registration
# -----------------------------------------
irds.register(docs, docs_ar, docs_zh, docs_en, docs_fr, docs_de, docs_it, docs_ja, docs_ko,
              docs_pt, docs_es, docs_catb)
