"""NanoBEIR — a small, fast preview of the headline BEIR evaluation suite.

Each of the 13 benchmarks here is a few dozen queries and a few thousand
docs sampled from its full BEIR counterpart (see ``beir.py``), published as
Parquet files on the HuggingFace Hub by zeta-alpha-ai -- convenient for quick
sanity checks where downloading/parsing the full BEIR corpora isn't worth it.

Reuses v1's ``NanoBeirDocs``/``NanoBeirQueries``/``NanoBeirQrels`` handler
classes as v2 ``parser=`` wrappers, same principle as ``beir.py``: they only
ever call ``.path()`` on what they're given, and a v2 File's ``.path()``
satisfies that identically to a v1 download.

Unlike full BEIR, every benchmark here is a flat docs+queries+qrels triple --
no multi-split derivation needed, since each Nano variant already *is* one
fixed sample.
"""
from ir_datasets.datasets.nano_beir import (
    NanoBeirDocs as _V1NanoBeirDocs, NanoBeirQrels as _V1NanoBeirQrels,
    NanoBeirQueries as _V1NanoBeirQueries,
)
from ir_datasets.formats import GenericDoc, GenericQuery

from ir_datasets.v2 import Benchmark, Docs, Qrels, Queries, Resource, Suite, irds
from ir_datasets.v2.formats import Parser

CITATION = ('zeta-alpha-ai/NanoBEIR (huggingface.co/collections/zeta-alpha-ai/nanobeir); '
           'see beir-suite for the full-size benchmarks these sample from')

#: v1 id -> {docs, queries, qrels: (url, md5, size)}, transcribed from
#: ir_datasets/etc/downloads.json's "nano-beir" entries.
SOURCES = {
    'arguana': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna/resolve/main/corpus/train-00000-of-00001.parquet', 'ee563dc09a91032b494f469b6c807a9c', 2207566),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna/resolve/main/queries/train-00000-of-00001.parquet', '5ad0a26fad750dc74eae00ab753731b8', 45590),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna/resolve/main/qrels/train-00000-of-00001.parquet', '5fab96653b0455879606e82071ef4b21', 3439),
    },
    'climate-fever': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER/resolve/main/corpus/train-00000-of-00001.parquet', '97a57296bbb7e48ffc8b763c4b2a188b', 3331342),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER/resolve/main/queries/train-00000-of-00001.parquet', '0f4d43544ae8755e0557f89be3ae7e74', 7569),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER/resolve/main/qrels/train-00000-of-00001.parquet', '3042ccd6054266951ac1338c57a2bb85', 3770),
    },
    'dbpedia-entity': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia/resolve/main/corpus/train-00000-of-00001.parquet', 'cee2bc11f2b84fe636b5f219246221e0', 1479972),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia/resolve/main/queries/train-00000-of-00001.parquet', 'ec7d771ba38310699a5b323aa693febe', 3546),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia/resolve/main/qrels/train-00000-of-00001.parquet', '8a9fa7e95c276d810ddbb53b8054801d', 23171),
    },
    'fever': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER/resolve/main/corpus/train-00000-of-00001.parquet', '0b052fca4b4c975088bc2f01d7f2b749', 3930640),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER/resolve/main/queries/train-00000-of-00001.parquet', '43122d82b3df866afad1f37a28e6ba60', 3881),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER/resolve/main/qrels/train-00000-of-00001.parquet', 'a25763bbae313d39668c37dbadb70cd3', 2730),
    },
    'fiqa': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018/resolve/main/corpus/train-00000-of-00001.parquet', '873e5dcf7920fc8c0ff210cf06907607', 2532554),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018/resolve/main/queries/train-00000-of-00001.parquet', '8b25da8b2d2e40a483d7a782a048a950', 4249),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018/resolve/main/qrels/train-00000-of-00001.parquet', 'f22d89ef6658379498bc92e92cf92d26', 2631),
    },
    'hotpotqa': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA/resolve/main/corpus/train-00000-of-00001.parquet', 'da3d14375223d0e5dad172acdbb1c332', 1247690),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA/resolve/main/queries/train-00000-of-00001.parquet', 'c1f13d4ed9f88a89285e39d9e13f5c38', 6431),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA/resolve/main/qrels/train-00000-of-00001.parquet', '3ed8424185156af1fe4c55aac84bce92', 3424),
    },
    'msmarco': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO/resolve/main/corpus/train-00000-of-00001.parquet', '15ab2a1d3674e562d088e64dc5781373', 1115821),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO/resolve/main/queries/train-00000-of-00001.parquet', '9ab46c94f88bfed1e338dc0887568f03', 3219),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO/resolve/main/qrels/train-00000-of-00001.parquet', 'c72caacf4bc04efa8af6b2ac2944ea92', 2095),
    },
    'nfcorpus': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus/resolve/main/corpus/train-00000-of-00001.parquet', 'deb02538a3b030843a8ccca212dc6a3c', 2462576),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus/resolve/main/queries/train-00000-of-00001.parquet', '6578fc992124bb997e32bd1192c1b475', 2786),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus/resolve/main/qrels/train-00000-of-00001.parquet', '402762544651e0fc6df10cd76e2cbfae', 16342),
    },
    'nq': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ/resolve/main/corpus/train-00000-of-00001.parquet', 'aaf73189d2b8d0de039dc2619d7a29ef', 1776094),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ/resolve/main/queries/train-00000-of-00001.parquet', '87de29b17a669fc61d1f192bd5099887', 3658),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ/resolve/main/qrels/train-00000-of-00001.parquet', 'fc2b183187c4f99e0939b55dc11b1908', 2020),
    },
    'quora': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval/resolve/main/corpus/train-00000-of-00001.parquet', 'a8cd6d41cbeb6bda2a3ed90526de4521', 229172),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval/resolve/main/queries/train-00000-of-00001.parquet', '590ed67697e12f21fafe436c07e6f772', 3918),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval/resolve/main/qrels/train-00000-of-00001.parquet', '650ca21cef9a73b83e3845c93e3a230a', 2266),
    },
    'scidocs': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS/resolve/main/corpus/train-00000-of-00001.parquet', '60f9f2a7206f0e8a0fc185df2b7ab00b', 1271482),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS/resolve/main/queries/train-00000-of-00001.parquet', '940e2220bc9667138774a36f6c771c97', 6874),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS/resolve/main/qrels/train-00000-of-00001.parquet', 'e21c1514761b06ac6bba5a8f958d8803', 14043),
    },
    'scifact': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact/resolve/main/corpus/train-00000-of-00001.parquet', '9f82bfc70f9b7d7f0275fb6b8ee38876', 2408011),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact/resolve/main/queries/train-00000-of-00001.parquet', '0218d8ad9c0bd8671002f16edb497ff3', 5482),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact/resolve/main/qrels/train-00000-of-00001.parquet', '1155d79cf3854fa43c72c8458f34f744', 2082),
    },
    'webis-touche2020': {
        'docs': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020/resolve/main/corpus/train-00000-of-00001.parquet', 'fdea27196d1234ab4aa2c49c9a849840', 7265307),
        'queries': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020/resolve/main/queries/train-00000-of-00001.parquet', '0ba668cdae413f411644871f544a9d1b', 3348),
        'qrels': ('https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020/resolve/main/qrels/train-00000-of-00001.parquet', '4b3de9c60d59be85ae4483e1210caff9', 17746),
    },
}


class _NanoBeirDocsParser(Parser):
    name = 'NanoBeirDocs'

    def __init__(self, v1_id):
        self.v1_id = v1_id

    def build(self, source, node):
        return _V1NanoBeirDocs(self.v1_id, source, GenericDoc)


class _NanoBeirQueriesParser(Parser):
    name = 'NanoBeirQueries'

    def __init__(self, v1_id):
        self.v1_id = v1_id

    def build(self, source, node):
        return _V1NanoBeirQueries(self.v1_id, source, GenericQuery)


class _NanoBeirQrelsParser(Parser):
    name = 'NanoBeirQrels'

    def build(self, source, node):
        return _V1NanoBeirQrels(source, node.defs or {1: 'relevant'})


with irds.defaults(lang='en'):
    nano_benchmarks = {}

    for v1_id, files in SOURCES.items():
        docs_url, docs_md5, docs_size = files['docs']
        queries_url, queries_md5, queries_size = files['queries']
        qrels_url, qrels_md5, qrels_size = files['qrels']

        # All three are .parquet on the Hub -- named by extension, not a
        # generic "-file" suffix.
        docs_file = Resource(f'nano-beir-{v1_id}-docs.parquet',
                             sources=[docs_url], md5=docs_md5, size=docs_size)
        queries_file = Resource(f'nano-beir-{v1_id}-queries.parquet',
                                sources=[queries_url], md5=queries_md5, size=queries_size)
        qrels_file = Resource(f'nano-beir-{v1_id}-qrels.parquet',
                              sources=[qrels_url], md5=qrels_md5, size=qrels_size)

        docs = Docs(f'nano-beir-{v1_id}-docs', source=docs_file,
                   parser=_NanoBeirDocsParser(v1_id))
        queries = Queries(f'nano-beir-{v1_id}-queries', source=queries_file,
                          parser=_NanoBeirQueriesParser(v1_id))
        qrels = Qrels(f'nano-beir-{v1_id}-qrels', source=qrels_file,
                      defs={1: 'relevant'}, parser=_NanoBeirQrelsParser())

        name = f'nano-beir-{v1_id}'
        nano_benchmarks[name] = Benchmark(
            name, docs=docs, queries=queries, qrels=qrels, citation=CITATION,
            desc=f'NanoBEIR: a small sample of BEIR\'s {v1_id} benchmark.')


# Registration
# -----------------------------------------
irds.register(*nano_benchmarks.values())
irds.register(Suite('nano-beir', benchmarks=list(nano_benchmarks.values()),
                    citation=CITATION,
                    desc="A small, fast preview of BEIR's headline benchmarks "
                         '(13 of 14) -- for quick sanity checks, not for '
                         'reporting comparable numbers.'))

# Aliases (old ir-datasets ID mapping)
# -----------------------------------------
irds.alias({f'nano-beir/{v1_id}': f'nano-beir-{v1_id}' for v1_id in SOURCES})
