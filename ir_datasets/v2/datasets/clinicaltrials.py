"""ClinicalTrials.gov -- a v2 dataset family (corpus for TREC Precision
Medicine's clinical-trials sub-task and TREC Clinical Trials; the benchmarks
themselves live in ``trec_pm.py``/``trec_ct.py`` -- see those modules'
docstrings for why they're no longer bundled here).

Reuses v1's ``ClinicalTrialsDocs`` handler class directly as a v2 ``parser=``
wrapper, same "thin layer over v1 machinery" every other v2 family uses (see
``beir.py``/``car.py``) -- it only ever calls ``.stream()`` on what it's
given, once per source in its list, and a list of v2 Source pipelines
satisfies that the same way a list of v1 download pipelines does. Each of
the three corpus snapshots (2017/2019/2021) already gets its own docstore
path from v1 (keyed by the ``name`` passed to ``ClinicalTrialsDocs``), so --
unlike ``car.py`` -- no subclassing is needed to avoid a path collision.
"""
from ir_datasets.datasets.clinicaltrials import ClinicalTrialsDocs as _V1ClinicalTrialsDocs
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser


class _ClinicalTrialsDocsParser(Parser):
    name = 'ClinicalTrialsDocs'

    def __init__(self, docstore_name, compress_format='tgz'):
        self.docstore_name = docstore_name
        self.compress_format = compress_format

    def build(self, source, node):
        sources = source if isinstance(source, (list, tuple)) else [source]
        return _V1ClinicalTrialsDocs(self.docstore_name, sources,
            compress_format=self.compress_format, count_hint=node.count_hint)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    docs_2017_file = Resource('clinicaltrials-2017-docs.tar.gz',
        sources=['https://bionlp.nlm.nih.gov/trec2017precisionmedicine/clinicaltrials_xml.tar.gz'],
        md5='e5d333ceed0cbbbe513504c96148ab1a',
        size=724_731_456,
    )
    docs_2019_files = [
        Resource(f'clinicaltrials-2019-docs-{i}.tar.gz',
            sources=[f'http://www.trec-cds.org/clinical_trials.{i}.tar.gz'],
            md5=md5, size=size)
        for i, (md5, size) in enumerate([
            ('d57fbafa63520c45faceedec3de801b7', 277_495_873),
            ('d32a632fc72c68d63309732af667b1ee', 266_267_182),
            ('055980d685164cb1daefa2b4cc1e9a2f', 242_899_908),
            ('fa8e76c85ab7204c0294f03fa76f6ed2', 218_698_307),
        ])
    ]
    docs_2021_files = [
        Resource(f'clinicaltrials-2021-docs-part{i}.zip',
            sources=[f'http://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part{i}.zip'],
            md5=md5, size=size)
        for i, (md5, size) in enumerate([
            ('e12eb9a0d21452503b0ef8874c69f490', 382_792_518),
            ('f6986125506434887a162f144ca4d9a2', 378_478_271),
            ('9b7fb528b22edfcf4535154cc3d98111', 375_998_752),
            ('4fd98d209e7b62cee87af211c0c281f6', 360_825_058),
            ('a747f09ac5d4f3cd0cc75957ad9f32d8', 296_625_845),
        ], start=1)
    ]

    # Tables
    # -----------------------------------------
    docs_2017 = DocTable('clinicaltrials-2017-docs',
        source=docs_2017_file,
        parser=_ClinicalTrialsDocsParser('2017'),
        count_hint=241_006,
    )
    docs_2019 = DocTable('clinicaltrials-2019-docs',
        source=docs_2019_files,
        parser=_ClinicalTrialsDocsParser('2019'),
        count_hint=306_238,
    )
    docs_2021 = DocTable('clinicaltrials-2021-docs',
        source=docs_2021_files,
        parser=_ClinicalTrialsDocsParser('2021', compress_format='zip'),
        count_hint=375_580,
    )


# Registration
# -----------------------------------------
irds.register(docs_2017, docs_2019, docs_2021)
