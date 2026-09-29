"""ClueWeb12 -- a v2 dataset family (the CMU/Lemur ClueWeb 2012 web crawl,
~733M English-only pages).

The corpus is LDC/NIST-gated and distributed on hard drives, not
automatically downloadable -- same shape as ``disks45.py``/``clueweb09.py``:
a ``Directory`` Resource pointed at by ``Source.external()``, at
``<home>/external/clueweb12`` (old v1 location ``<home>/clueweb12/corpus``
still read as a fallback).
A separately downloadable "chk" archive (per-WARC-file record-offset
checkpoints, used for random access / seeking within ``docs.lookup``) is a
real downloadable ``Resource``, extracted via v1's ``TarExtractAll`` -- same
pattern as ``clueweb09.py``. Unlike ClueWeb09, there is no per-language
split here: all of ClueWeb12 is English, so v1's own ``ClueWeb12Docs``
hardcodes ``lang='en'`` and this module does not thread a ``dirs=`` filter
through it the way ``clueweb09.py`` does.

Reuses v1's ``ClueWeb12Docs`` (a ``WarcDocs`` subclass) directly as a v2
``parser=`` wrapper, and v1's ``ClueWeb12b13Extractor`` unmodified for the
"b13" subset below -- ``WarcDocs`` has no general-purpose v2 format node, so
both go through a local ``Parser``, same "thin layer over v1 machinery"
pattern as every other migrated family.

**The b13 subset (``clueweb12-b13-docs``) requires an external tool run by
the user, same spirit as ``aol_ia.py``'s documented external-tool
prerequisite.** v1's ``ClueWeb12b13Extractor`` does not stream/download the
b13 subset the normal way: its ``.path(force=True)`` either (a) returns an
already-materialized ``<docs_path>-b13`` directory if the user has already
built or linked one there, or (b) raises a ``RuntimeError`` with instructions
telling the user to run a Java extraction tool themselves
(``ClueWeb12-CreateB13Dataset.jar``) against the full corpus:

    java -jar <jar path> <docs_path>/ <docs_path>-b13/

That jar is not itself hand-obtained -- it is a real downloadable Resource
(``cw12b-info``, ``ClueWeb12-CreateB13.tgz`` from lemurproject.org), just one
whose tar member is bz2-compressed and needs decompressing before use. v1's
own construction is ``Bz2Extract(Cache(TarExtract(dlc['cw12b-info'], '.../
CreateClueWeb12B13Dataset.jar'), base_path/'CreateClueWeb12B13Dataset.jar'))``
-- ``Cache`` wraps the *raw* (still bz2-compressed) tar member so it survives
across runs, and ``Bz2Extract`` decompresses it when read as a stream.
Crucially, v1's ``ClueWeb12b13Extractor.path()`` calls ``.path()`` on that
whole chain, not ``.stream()`` -- and ``Bz2Extract.__getattr__`` delegates
any attribute it doesn't define itself (including ``path``) straight through
to the thing it wraps, so ``.path()`` here actually returns the *Cache*'s
path (the raw member on disk), bypassing decompression entirely. b13's v2
equivalent, ``b13_extract_jar`` below, reproduces this exactly with
``Readable`` pipeline ops in the same nesting order --
``.member(...).cache(path).bunzip2()`` (member extraction first, cached to
disk, *then* wrapped in bunzip2) -- because ``Readable``'s own ``_Pipe.path()``
has the identical "delegate to the wrapped step if this step has no ``path``
of its own" behavior, so the same bypass happens here too. Getting the
nesting order backwards (``.bunzip2()`` before ``.cache()``) would decompress
before caching and no longer match v1's on-disk layout or its ``.path()``
short-circuit.

Doc counts (from ``ir_datasets/etc/metadata.json``): full corpus
733,019,372; b13 subset 52,343,021.

NTCIR-WWW (``ntcir_www.py``), TREC Misinfo 2019 (``trec_misinfo.py``) and
CLEF eHealth (``clef_ehealth.py``) all judge ``clueweb12-b13-docs`` and live
in their own files, importing ``docs_b13`` from here by reference, same
cross-file pattern as ``disks45.py``/``trec_adhoc.py``. TREC Web Track
2013/2014 (ad hoc + diversity) judge the *full* ``clueweb12-docs`` collection
and are **not** defined here -- see ``trec_web.py``, which imports ``docs``
from here the same way. Touché's argsme-backed clueweb12 tasks
(``clueweb12/touche-2020-task-2`` etc.) are not yet migrated (``touche.py``
itself isn't migrated) and are not this file's concern either.
"""
import ir_datasets
from ir_datasets.datasets.clueweb12 import ClueWeb12Docs as _V1ClueWeb12Docs
from ir_datasets.datasets.clueweb12 import ClueWeb12b13Extractor
from ir_datasets.util import TarExtractAll
from ir_datasets.v2 import Directory, DocTable, Resource, Source, irds
from ir_datasets.v2.formats import Parser

NAME = 'clueweb12'

DUA = ("ClueWeb12 is distributed on hard drives by CMU "
       "<https://lemurproject.org/clueweb12/>; your organization may need to "
       "file an organizational agreement and pay a fee to CMU.")

DATA_ACCESS = [
    "ClueWeb12 is available on hard drives from CMU: <https://lemurproject.org/clueweb12/>.\nYour organization may already have a copy, in which case you may only need to complete a new \"Individual Agreement\". Otherwise your organization must file the \"Organizational agreement\" and pay a fee to CMU; the data are shipped on hard drives.",
    "Gather the ClueWeb12 source directories into one directory; it should contain directories ClueWeb12_00 through ClueWeb12_19.",
]

DOCS_DEFAULT_PATH = 'clueweb12'
DOCS_OLD_LOCATION = f'{NAME}/corpus'
BASE_PATH = ir_datasets.util.home_path() / NAME


class _ClueWeb12DocsParser(Parser):
    """The full collection: a plain Directory tree plus a downloadable chk
    checkpoint archive -- same shape as ``clueweb09.py``'s parser."""
    name = 'ClueWeb12Docs'

    def build(self, source, node):
        docs_source, chk_source = source
        chk = TarExtractAll(chk_source, BASE_PATH / 'corpus.chk')
        return _V1ClueWeb12Docs(docs_source, chk)


class _ClueWeb12b13DocsParser(Parser):
    """The b13 subset -- reuses v1's ``ClueWeb12b13Extractor`` unmodified
    (see module docstring for why its ``.path()`` needs an external tool, and
    why ``b13_extract_jar``'s pipeline nesting order matters)."""
    name = 'ClueWeb12Docs(b13)'

    def build(self, source, node):
        docs_source, jar_source = source
        extractor = ClueWeb12b13Extractor(docs_source, jar_source)
        return _V1ClueWeb12Docs(extractor)


# Files
# -----------------------------------------
docs_file = Directory('clueweb12.dir',
    sources=[Source.external(DOCS_DEFAULT_PATH, old_locations=[DOCS_OLD_LOCATION], instructions=DATA_ACCESS)],
    dua=DUA,
)
docs_chk_file = Resource('clueweb12-chk.tar.gz',
    sources=['https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/ir-datasets/clueweb12/clueweb12-source-chk.tar.gz'],
    hash='md5:fb92d1f8ed1436839313d2eb47f628a5',
    size=3_883_120_643,
)
cw12b_info_file = Resource('clueweb12-cw12b-info.tgz',
    sources=['http://lemurproject.org/clueweb12-CreateB13.tgz'],
    hash='md5:8175ce74a97e46be80c2127d965da200',
    size=1_310_407_043,
)
# See module docstring: member -> cache -> bunzip2, matching v1's
# Bz2Extract(Cache(TarExtract(...))) nesting (outermost wraps innermost) and
# its .path()-bypasses-decompression behavior exactly.
b13_extract_jar = (cw12b_info_file
    .member('ClueWeb12-CreateB13/software/CreateClueWeb12B13Dataset.jar')
    .cache(BASE_PATH / 'CreateClueWeb12B13Dataset.jar')
    .bunzip2())

# Tables
# -----------------------------------------
docs = DocTable('clueweb12',
    source=[docs_file, docs_chk_file], parser=_ClueWeb12DocsParser(),
    lang='en', count_hint=733_019_372)

docs_b13 = DocTable('clueweb12-b13',
    source=[docs_file, b13_extract_jar], parser=_ClueWeb12b13DocsParser(),
    lang='en', count_hint=52_343_021)


# Registration
# -----------------------------------------
irds.register(docs, docs_b13)
