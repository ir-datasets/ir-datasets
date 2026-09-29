"""WikIR -- a v2 dataset family.

Seven Wikipedia-derived IR benchmarks (five languages, with English at three
sizes) from the WikIR/MLWikIR toolkits. Each variant is one self-contained
zip: a shared ``documents.csv`` corpus plus, per split
(train/valid/test), a ``queries.csv``, a TREC-format ``qrels`` file,
and a pre-computed BM25 run (``BM25.res``) -- so this is the first v2 family
whose benchmarks carry a ``scoreddocs`` facet.

Reuses v1's ``CsvDocs``/``CsvQueries`` handler classes directly as v2
``parser=`` wrappers (the usual "thin layer over v1 machinery"); qrels/
scoreddocs need no wrapper at all since ``TrecQrels``/``TrecScoredDocs``
already exist as v2 format nodes. Zip members are read straight off each
variant's own downloaded zip (``.zip_member(...)``, no ``.cache()`` needed --
the zip itself is already the one thing fetched/cached, and extracting one
member from a local zip is cheap).

v1 keys everything by a short version-y code (``en1k``, ``fr14k``, ...); kept
verbatim here since it's already the natural short name -- ``wikir-{code}``,
not something derived per-language that would collide (English alone has three
sizes).
"""
from ir_datasets.formats import CsvDocs as _V1CsvDocs, CsvQueries as _V1CsvQueries

from ir_datasets.v2 import (
    Benchmark, DocTable, Parser, QueryTable, Resource, TrecQrels, TrecScoredDocs,
    irds,
)

CITATION = 'dblp:conf/lrec/FrejSC20; dblp:conf/circle/FrejSC20a'

QRELS_DEFS = {
    2: "Query is the article title",
    1: "There is a link to the article with the query as its title in the first sentence",
    0: "Otherwise",
}

#: (code, zip's own top-level directory name, lang, url, md5, size) -- verbatim
#: from ir_datasets.datasets.wikir's ``sources`` list plus etc/downloads.json.
VARIANTS = [
    ('en1k', 'wikIR1k', 'en', 'https://zenodo.org/record/3565761/files/wikIR1k.zip',
     '554299bca984640cb283d6ba55753608', 164_995_559),
    ('en59k', 'wikIR59k', 'en', 'https://zenodo.org/record/3557342/files/wikIR59k.zip',
     'c9f7e646e022eea84e6f00e3870ca79b', 1_154_400_672),
    ('en78k', 'enwikIR', 'en', 'https://www.zenodo.org/record/3707606/files/enwikIR.zip',
     'e1a1f7678523032e0be5fedaed6c0740', 4_234_761_118),
    ('ens78k', 'enwikIRS', 'en', 'https://www.zenodo.org/record/3707238/files/enwikIRS.zip',
     '8fd2e530ec9dfd17f3b305ec23122b55', 4_245_785_781),
    ('fr14k', 'FRwikIR14k', 'fr', 'https://zenodo.org/record/3569718/files/FRwikIR14k.zip',
     '0bf8a8965b1a550ad3604a9ddd5c6bbe', 331_209_361),
    ('es13k', 'ESwikIR13k', 'es', 'https://zenodo.org/record/3569724/files/ESwikIR13k.zip',
     '4847eeffbf261d3877da86f5ccae4e43', 299_523_201),
    ('it16k', 'ITwikIR16k', 'it', 'https://zenodo.org/record/3569732/files/ITwikIR16k.zip',
     'e9c5b81c9df6fdc0e2986fa8ffb8ff12', 248_875_419),
]

#: (v1 split name, short node-name suffix)
SPLITS = (('training', 'train'), ('validation', 'valid'), ('test', 'test'))


class _WikirDocsParser(Parser):
    name = 'CsvDocs'

    def build(self, source, node):
        return _V1CsvDocs(source, lang=node.lang, count_hint=node.count_hint,
                          docstore_path=str(node.docstore_path))


class _WikirQueriesParser(Parser):
    name = 'CsvQueries'

    def build(self, source, node):
        return _V1CsvQueries(source, lang=node.lang)


_benchmarks = []

for _code, _zip_dir, _lang, _url, _md5, _size in VARIANTS:
    _zip = Resource(f'wikir-{_code}.zip', sources=[_url], hash=f'md5:{_md5}', size=_size)

    _docs = DocTable(f'wikir-{_code}-docs',
        source=_zip.zip_member(f'{_zip_dir}/documents.csv'),
        parser=_WikirDocsParser(), lang=_lang,
        desc=f'The WikIR {_code} document corpus.')

    for _split, _suffix in SPLITS:
        _name = f'wikir-{_code}-{_suffix}'
        _queries = QueryTable(f'{_name}-queries',
            source=_zip.zip_member(f'{_zip_dir}/{_split}/queries.csv'),
            parser=_WikirQueriesParser(), lang=_lang)
        _qrels = TrecQrels(f'{_name}-qrels',
            source=_zip.zip_member(f'{_zip_dir}/{_split}/qrels'),
            defs=QRELS_DEFS)
        _scoreddocs = TrecScoredDocs(f'{_name}-scoreddocs',
            source=_zip.zip_member(f'{_zip_dir}/{_split}/BM25.res'))
        _benchmarks.append(Benchmark(_name,
            docs=_docs, queries=_queries, qrels=_qrels, scoreddocs=_scoreddocs,
            citation=CITATION,
            desc=f'WikIR {_code}, {_split} split. Scoreddocs are the '
                 'provided BM25 run.'))


# Registration
# -----------------------------------------
irds.register(*_benchmarks)

