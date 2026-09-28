"""Medline -- a v2 dataset family (corpus for TREC Genomics 2004/2005 and
TREC Precision Medicine 2017/2018 abstracts sub-task).

Reuses v1's ``MedlineDocs``/``AacrAscoDocs``/``ConcatDocs`` handler classes
directly as v2 ``parser=`` wrappers, same "thin layer over v1 machinery"
pattern as ``clinicaltrials.py``/``highwire.py``. The 2004 corpus is four
gzip-compressed XML chunks that v1 concatenates into one logical stream
(``.gunzip()`` on each Resource reproduces v1's ``GzipExtract`` wrapping);
the 2017 corpus is five gzip-tarred XML parts plus an AACR/ASCO-abstracts tar
concatenated after them (v1's ``ConcatDocs``), reassembled the same way.

TREC Genomics 2004/2005's benchmarks live in ``trec_genomics.py`` (which
also covers highwire.py's 2006/2007 years of the same track). TREC
Precision Medicine 2017/2018's benchmarks live in ``trec_pm.py`` (which also
covers clinicaltrials.py's clinical-trials sub-task for those years) -- both
moved out of here per the "track spans multiple corpora" convention
trec_web.py/trec_tb.py/trec_mq.py established, only the corpus Tables
(``docs_2004``/``docs_2017``) stay in this file.
"""
from ir_datasets.datasets.medline import AacrAscoDocs as _V1AacrAscoDocs
from ir_datasets.datasets.medline import ConcatDocs as _V1ConcatDocs
from ir_datasets.datasets.medline import MedlineDocs as _V1MedlineDocs
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser


class _Medline2004DocsParser(Parser):
    name = 'MedlineDocs2004'

    def build(self, source, node):
        return _V1MedlineDocs('2004', source, count_hint=node.count_hint)


class _Medline2017DocsParser(Parser):
    name = 'MedlineDocs2017'

    def build(self, source, node):
        *medline_parts, aacr_asco_source = source
        aacr_asco = _V1AacrAscoDocs(aacr_asco_source)
        medline = _V1MedlineDocs('2017', medline_parts)
        return _V1ConcatDocs([aacr_asco, medline], count_hint=node.count_hint)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    docs_2004_files = [
        Resource(f'medline-2004-docs-{part}.gz',
            sources=[f'https://dmice.ohsu.edu/trec-gen/data/2004/XML/2004_TREC_XML_MEDLINE_{part.upper()}.gz'],
            md5=md5, size=size)
        for part, (md5, size) in [
            ('a', ('7858e5b908c25b88e30965b770e9780f', 579_470_012)),
            ('b', ('d4f8b510716d71612dc84129b7bc86a8', 623_531_586)),
            ('c', ('155c77b6f75b549a810863dae03058d1', 593_454_144)),
            ('d', ('cb1570f0212f8b737c757b6177788f36', 599_821_183)),
        ]
    ]
    docs_2017_parts_files = [
        Resource(f'medline-2017-docs-part{i}.tar.gz',
            sources=[f'https://bionlp.nlm.nih.gov/trec2017precisionmedicine/medline_xml.part{i}.tar.gz'],
            md5=md5, size=size)
        for i, (md5, size) in enumerate([
            ('04d14a46af586faf9306580291758c29', 5_257_751_264),
            ('19740bcde4e5e3bcfc583b347cd59d17', 5_257_075_322),
            ('7e7d3cfb452c6e4260704f0a6cc7932e', 5_249_034_853),
            ('1b22d0932d319d127be326358435c5e5', 5_245_054_502),
            ('d71b9bb9e11d017f3f77ffe47dbf8aa9', 1_187_092_702),
        ], start=1)
    ]
    docs_2017_aacr_asco_file = Resource('medline-2017-aacr-asco-docs.tar.gz',
        sources=['https://bionlp.nlm.nih.gov/trec2017precisionmedicine/extra_abstracts.tar.gz'],
        md5='d91bb4ca9b50cbbd5986bb5c43082afb',
        size=61_150_087,
    )

    # Tables
    # -----------------------------------------
    docs_2004 = DocTable('medline-2004-docs',
        source=[f.gunzip() for f in docs_2004_files],
        parser=_Medline2004DocsParser(),
        count_hint=3_672_808,
    )
    docs_2017 = DocTable('medline-2017-docs',
        source=[*docs_2017_parts_files, docs_2017_aacr_asco_file],
        parser=_Medline2017DocsParser(),
        count_hint=26_740_025,
    )


# Registration
# -----------------------------------------
irds.register(docs_2004, docs_2017)
