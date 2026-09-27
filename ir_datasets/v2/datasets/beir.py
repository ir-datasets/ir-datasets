"""BEIR — a suite of heterogeneous zero-shot retrieval benchmarks.

BEIR is the worked example the ``Suite`` node type was designed around: many
independent corpora (different domains, different doc/query schemas), each
with its own official evaluation split, grouped under one name.

Reuses v1's ``BeirDocs``/``BeirQueries``/``BeirQrels`` handler classes
directly as v2 ``parser=`` wrappers, rather than re-deriving JSONL parsing
from scratch -- the same "thin layer over v1 machinery" every other v2
family uses. Those classes only ever call ``.stream()`` on what they're
given, and a v2 ``Source`` pipeline (``resource.zip_member(...)``) satisfies
that same interface a v1 download pipeline does, so nothing about them needs
to change. A pleasant side effect: their docstore/cache paths are computed
from the *same* v1-module-level constants as before, so an existing v1 BEIR
download is read in place, not re-fetched.

Fidelity kept from v1: the per-domain typed doc/query records (``BeirSciDoc``
carries ``authors``/``year``/citations for SciDocs, ``BeirCqaDoc`` carries
``tags`` for CQADupStack, ...) are reused as-is rather than collapsed to one
generic shape -- BEIR's whole point is that these corpora are heterogeneous,
so flattening their schemas would lose exactly what the suite demonstrates.

Multi-split families (msmarco, nfcorpus, hotpotqa, fiqa, quora,
dbpedia-entity, fever, scifact) have one *derived* Benchmark per split, whose
queries are the full, unfiltered query set filtered down to the ids the
split's own qrels file judges (``Filter(query_ids=ids_of(...),
mode='include')``) and whose qrels are that split's real qrels table, set
explicitly rather than derived. The full query set has no single "the" qrels
for these families, so it is not itself a usable benchmark and is not
registered -- it lives only as an unregistered parent (see ``bare`` below)
for the splits to derive from. Single-split families (trec-covid, nq,
arguana, ...) and CQADupStack's 12 sub-forums need no derivation: one flat
Benchmark each.
"""
from ir_datasets.datasets.beir import (
    BeirCordDoc, BeirCovidQuery, BeirCqaDoc, BeirCqaQuery, BeirDocs as _V1BeirDocs,
    BeirQrels as _V1BeirQrels, BeirQueries as _V1BeirQueries, BeirSciDoc, BeirSciQuery,
    BeirTitleDoc, BeirTitleUrlDoc, BeirToucheDoc, BeirToucheQuery, BeirUrlQuery,
)
from ir_datasets.formats import GenericDoc, GenericQuery

from ir_datasets.v2 import Benchmark, DocTable, Filter, QrelTable, QueryTable, Resource, Suite, ids_of, irds
from ir_datasets.v2.formats import Parser

CITATION = 'Thakur et al., 2021, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (arxiv:2104.08663)'

#: (v1 id, qrel splits, doc record type, query record type) -- transcribed
#: verbatim from ir_datasets.datasets.beir's own ``benchmarks`` dict.
BENCHMARKS = [
    ('msmarco', ['train', 'dev', 'test'], GenericDoc, GenericQuery),
    ('trec-covid', ['test'], BeirCordDoc, BeirCovidQuery),
    ('nfcorpus', ['train', 'dev', 'test'], BeirTitleUrlDoc, BeirUrlQuery),
    ('nq', ['test'], BeirTitleDoc, GenericQuery),
    ('hotpotqa', ['train', 'dev', 'test'], BeirTitleUrlDoc, GenericQuery),
    ('fiqa', ['train', 'dev', 'test'], GenericDoc, GenericQuery),
    ('arguana', ['test'], BeirTitleDoc, GenericQuery),
    ('webis-touche2020', ['test'], BeirToucheDoc, BeirToucheQuery),
    ('webis-touche2020/v2', ['test'], BeirToucheDoc, BeirToucheQuery),
    ('quora', ['dev', 'test'], GenericDoc, GenericQuery),
    ('dbpedia-entity', ['dev', 'test'], BeirTitleUrlDoc, GenericQuery),
    ('scidocs', ['test'], BeirSciDoc, BeirSciQuery),
    ('fever', ['train', 'dev', 'test'], BeirTitleDoc, GenericQuery),
    ('climate-fever', ['test'], BeirTitleDoc, GenericQuery),
    ('scifact', ['train', 'test'], BeirTitleDoc, GenericQuery),
]

#: v1 id -> (zip URL, md5, size). One File per benchmark's own zip; CQADupStack
#: (12 sub-forums sharing one zip) is handled separately below.
ZIP_SOURCES = {
    'msmarco': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip', '444067daf65d982533ea17ebd59501e4', 1082258632),
    'trec-covid': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip', 'ce62140cb23feb9becf6270d0d1fe6d1', 73876720),
    'nfcorpus': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip', 'a89dba18a62ef92f7d323ec890a0d38d', 2448432),
    'nq': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip', 'd4d3d2e48787a744b6f6e691ff534307', 498307926),
    'hotpotqa': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip', 'f412724f78b0d91183a0e86805e16114', 654025350),
    'fiqa': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip', '17918ed23cd04fb15047f73e6c3bd9d9', 17948027),
    'arguana': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip', '8ad3e3c2a5867cdced806d6503f29b99', 3773617),
    'webis-touche2020': ('https://macavaney.us/beir-webis-touche2020-v1.zip', '5ec7f8b18481fc2e9b3964ad1b22dd28', 227137373),
    'webis-touche2020/v2': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip', '46f650ba5a527fc69e0a6521c5a23563', 227132363),
    'quora': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip', '18fb154900ba42a600f84b839c173167', 15853968),
    'dbpedia-entity': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip', 'c2a39eb420a3164af735795df012ac2c', 639285131),
    'scidocs': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip', '38121350fc3a4d2f48850f6aff52e4a9', 142471588),
    'fever': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip', '5a818580227bfb4b35bb6fa46d9b6c03', 1236988269),
    'climate-fever': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip', '8b66f0a9126c521bae2bde127b4dc99d', 1228666652),
    'scifact': ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip', '5f7d1de60b170fc8027bb7898e2efca1', 2816079),
}

CQA_SUBFORUMS = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics',
                  'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
CQA_ZIP = ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip',
           '4e41456d7df8ee7760a7f866133bda78', 5343728040)

#: The 14 headline benchmarks -- the 13 NanoBEIR also mirrors, plus trec-covid
#: (the 14th of the commonly-reported "main" BEIR results, omitted from
#: NanoBEIR only because it's large). CQADupStack's 12 sub-forums are their
#: own nested suite (see registration below), not flattened in here.
SUITE_MEMBERS = [
    'beir-trec-covid', 'beir-nfcorpus-test', 'beir-nq', 'beir-hotpotqa-test',
    'beir-fiqa-test', 'beir-arguana', 'beir-webis-touche2020-v2', 'beir-quora-test',
    'beir-dbpedia-entity-test', 'beir-scidocs', 'beir-fever-test',
    'beir-climate-fever', 'beir-scifact-test', 'beir-msmarco-dev',
]


class _BeirDocsParser(Parser):
    name = 'BeirDocs'

    def __init__(self, v1_id, doc_type):
        self.v1_id = v1_id
        self.doc_type = doc_type

    def build(self, source, node):
        return _V1BeirDocs(self.v1_id, source, self.doc_type)


class _BeirQueriesParser(Parser):
    name = 'BeirQueries'

    def __init__(self, v1_id, query_type):
        self.v1_id = v1_id
        self.query_type = query_type

    def build(self, source, node):
        return _V1BeirQueries(self.v1_id, source, self.query_type)


class _BeirQrelsParser(Parser):
    name = 'BeirQrels'

    def build(self, source, node):
        return _V1BeirQrels(source, node.defs or {})


def _flat(v1_id):
    """'webis-touche2020/v2' -> 'webis-touche2020-v2' (flat v2 naming)."""
    return v1_id.replace('/', '-')


def _docs(v1_id, zip_resource, doc_type):
    zip_folder = v1_id.split('/')[0]  # the /v2 zip still unpacks to the v1 folder name
    return DocTable(f'beir-{_flat(v1_id)}-docs',
               source=zip_resource.zip_member(f'{zip_folder}/corpus.jsonl'),
               parser=_BeirDocsParser(v1_id, doc_type))


def _queries(v1_id, zip_resource, query_type):
    zip_folder = v1_id.split('/')[0]
    return QueryTable(f'beir-{_flat(v1_id)}-queries',
                   source=zip_resource.zip_member(f'{zip_folder}/queries.jsonl'),
                   parser=_BeirQueriesParser(v1_id, query_type))


def _qrels(v1_id, zip_resource, split=None):
    zip_folder = v1_id.split('/')[0]
    suffix = f'-{split}' if split else ''
    return QrelTable(f'beir-{_flat(v1_id)}{suffix}-qrels',
                source=zip_resource.zip_member(f'{zip_folder}/qrels/{split or "test"}.tsv'),
                parser=_BeirQrelsParser())


with irds.defaults(lang='en'):
    benchmarks = {}   # flat name -> Benchmark, for suite assembly + aliasing

    for v1_id, splits, doc_type, query_type in BENCHMARKS:
        zip_url, zip_md5, zip_size = ZIP_SOURCES[v1_id]
        zip_resource = Resource(f'beir-{_flat(v1_id)}.zip',
                                sources=[zip_url], md5=zip_md5, size=zip_size)
        docs = _docs(v1_id, zip_resource, doc_type)
        queries = _queries(v1_id, zip_resource, query_type)

        if len(splits) == 1:
            qrels = _qrels(v1_id, zip_resource, splits[0] if splits[0] != 'test' else None)
            name = f'beir-{_flat(v1_id)}'
            benchmarks[name] = Benchmark(name, docs=docs, queries=queries, qrels=qrels,
                                         citation=CITATION,
                                         desc=f'BEIR: {v1_id}.')
        else:
            # Not registered: a private parent for the per-split Benchmarks
            # below to inherit docs/queries from (filtered by each split's
            # qrels) via derived_from. There is no single qrels for the full
            # query set, so this itself would not be a usable benchmark --
            # see the per-split ones instead.
            bare = Benchmark(f'beir-{_flat(v1_id)}', docs=docs, queries=queries,
                             citation=CITATION)
            for split in splits:
                split_name = f'beir-{_flat(v1_id)}-{split}'
                split_qrels = _qrels(v1_id, zip_resource, split)
                benchmarks[split_name] = Benchmark(
                    split_name, derived_from=bare, qrels=split_qrels,
                    # ids_of() resolves through the top-level graph, so it
                    # needs a qualified name -- not just split_qrels.name.
                    filter=Filter(query_ids=ids_of(f'irds:{split_qrels.name}'), mode='include'),
                    citation=CITATION,
                    desc=f'BEIR: {v1_id}, {split} split.')

    # CQADupStack: 12 sub-forums sharing one zip, each single-split.
    cqa_url, cqa_md5, cqa_size = CQA_ZIP
    cqa_zip = Resource('beir-cqadupstack.zip', sources=[cqa_url], md5=cqa_md5, size=cqa_size)
    for sub in CQA_SUBFORUMS:
        v1_id = f'cqadupstack/{sub}'
        docs = DocTable(f'beir-cqadupstack-{sub}-docs',
                   source=cqa_zip.zip_member(f'cqadupstack/{sub}/corpus.jsonl'),
                   parser=_BeirDocsParser(v1_id, BeirCqaDoc))
        queries = QueryTable(f'beir-cqadupstack-{sub}-queries',
                          source=cqa_zip.zip_member(f'cqadupstack/{sub}/queries.jsonl'),
                          parser=_BeirQueriesParser(v1_id, BeirCqaQuery))
        qrels = QrelTable(f'beir-cqadupstack-{sub}-qrels',
                      source=cqa_zip.zip_member(f'cqadupstack/{sub}/qrels/test.tsv'),
                      parser=_BeirQrelsParser())
        name = f'beir-cqadupstack-{sub}'
        benchmarks[name] = Benchmark(name, docs=docs, queries=queries, qrels=qrels,
                                     citation=CITATION, desc=f'BEIR: cqadupstack/{sub}.')


# Registration
# -----------------------------------------
irds.register(*benchmarks.values())

cqa_suite = Suite('beir-cqadupstack',
                  benchmarks=[benchmarks[f'beir-cqadupstack-{sub}'] for sub in CQA_SUBFORUMS],
                  citation=CITATION,
                  desc="CQADupStack: BEIR's 12 StackExchange sub-forums, "
                       'aggregated from a single shared download -- its own '
                       'suite since it is itself commonly reported as one '
                       'sub-benchmark group, and nested into the main BEIR '
                       'suite below rather than flattened into it.')
irds.register(cqa_suite)

irds.register(Suite('beir', benchmarks=[benchmarks[n] for n in SUITE_MEMBERS],
                    suites=[cqa_suite],
                    citation=CITATION,
                    desc='The BEIR evaluation suite: 14 headline zero-shot '
                         'retrieval benchmarks across heterogeneous domains, '
                         "plus CQADupStack's 12 StackExchange sub-forums "
                         '(nested as the beir-cqadupstack suite).'))

# Aliases (old ir-datasets ID mapping)
# -----------------------------------------
irds.alias({f'beir/{v1_id}': benchmarks[f'beir-{_flat(v1_id)}'].name
           for v1_id, splits, *_ in BENCHMARKS if len(splits) == 1})
irds.alias({f'beir/{v1_id}/{split}': benchmarks[f'beir-{_flat(v1_id)}-{split}'].name
           for v1_id, splits, *_ in BENCHMARKS if len(splits) > 1 for split in splits})
irds.alias({f'beir/cqadupstack/{sub}': benchmarks[f'beir-cqadupstack-{sub}'].name
           for sub in CQA_SUBFORUMS})
