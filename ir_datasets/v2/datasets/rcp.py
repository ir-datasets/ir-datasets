"""RCP (Rubric-Calibrated Preferences) -- calibrated LLM relevance labels over
the NanoBEIR benchmarks.

Schmidt, Crisostomi, Lassance & Reimers (Cohere), "Rubric-Calibrated
Preferences: Cross-Query Calibration of LLM Judgments via Item Response
Theory" (arXiv 2609.35739). RCP labels each benchmark's 150-document
candidate pool with a calibrated pass probability in [0, 1] (the gain
RCP-nDCG uses) and a latent ability ``theta`` on a scale shared across
queries. It introduces no new documents or queries: each ``nano-beir-<task>-rcp``
benchmark is ``nano-beir-<task>``'s docs and queries with the RCP qrels.

The HF repo (``{REPO}``) ships its own copy of every task's corpus and
queries. Checked against ``nano-beir`` for all 13 tasks, they are identical
(same ids, same text, same title -- RCP's titles are all empty), so those
copies are not redefined here.

``relevance`` is the float gain, not an integer grade. Documents that are
human-positive but fall outside the judged pool have ``relevance = 0.0`` and
``theta = None``.

Only NanoBEIR is covered; the same HF org also publishes BRIGHT, TREC-DL and
ViDoRe v3 labels.
"""
from ir_datasets.v2 import Benchmark, Resource, Suite, irds
from ir_datasets.v2.datasets.nano_beir import nano_benchmarks
from ir_datasets.v2.formats import RcpQrels

CITATION = 'dblp:journals/corr/abs-2609-35739'

REPO = 'fabianschmidt-cohere/rcp-ndcg-nanobeir'
COMMIT = 'cb6c981c80a70b6a46f8b7ba3a60206a9fab5d8e'

#: nano-beir id -> (HF subset directory, (sha256, size) of its qrels.parquet),
#: from the Hub API at COMMIT.
SOURCES = {
    'arguana': ('NanoArguAnaRetrieval', {
        'qrels': ('sha256:c30bac3ff767f04138dbe54567accdde3d6b1cbf83096e8225bbd2ddeeddb16e', 172853),
    }),
    'climate-fever': ('NanoClimateFeverRetrieval', {
        'qrels': ('sha256:8dac44c58782868b1c23133ffc061d5baa50b79a274039ebadebd88db5cde4f1', 167939),
    }),
    'dbpedia-entity': ('NanoDBPediaRetrieval', {
        'qrels': ('sha256:bf0c1ba0d17332ea8525e2a323d716f0b96e7beb56e6029949bba644f0b3232f', 211876),
    }),
    'fever': ('NanoFEVERRetrieval', {
        'qrels': ('sha256:7a0bfecb95e4683d15511d9d847f986da5105594d42e64772d2a9b1848f6f844', 198166),
    }),
    'fiqa': ('NanoFiQA2018Retrieval', {
        'qrels': ('sha256:33aac8b4c177ab28ffcf2f6815107d6f8ee7ba87fdf7089bce0b872cd37f5684', 161829),
    }),
    'hotpotqa': ('NanoHotpotQARetrieval', {
        'qrels': ('sha256:d457432b14631eda13df994724f70ab6293b2ba93fb81cc65cc9a07f56a7cc88', 170739),
    }),
    'msmarco': ('NanoMSMARCORetrieval', {
        'qrels': ('sha256:4d52ef215cd73192e9ce6c1eb67a0929792dc761775ef6e6387eee6b602833a9', 170371),
    }),
    'nfcorpus': ('NanoNFCorpusRetrieval', {
        'qrels': ('sha256:bc652612ea875c2b40b6b1fc01930166acf4ce492bfe5940ccafc473d4ad653c', 157400),
    }),
    'nq': ('NanoNQRetrieval', {
        'qrels': ('sha256:073c6b1080bbb3b28792ff96f7f7af263e8355713d932e1dfa968153a5263dbc', 168721),
    }),
    'quora': ('NanoQuoraRetrieval', {
        'qrels': ('sha256:743d872a1eb5c8950813139738158d00f6bba1c1325daaa60d3f9017fd7fd712', 166563),
    }),
    'scidocs': ('NanoSCIDOCSRetrieval', {
        'qrels': ('sha256:5392651a4f67aaa9facbbff4ace3a23eb349fa85871e8175f68c7e69fdba776f', 187735),
    }),
    'scifact': ('NanoSciFactRetrieval', {
        'qrels': ('sha256:6f123db8153b9f11ae2edc742dd5d41bb04372ab7d275d17074fd5a2f085adc1', 160828),
    }),
    'webis-touche2020': ('NanoTouche2020Retrieval', {
        'qrels': ('sha256:dc7cb5acc22216276f08d2225da6588e809aed9307ffb4af44adab573b0bdd0a', 190498),
    }),
}


rcp_benchmarks = {}

for v1_id, (subset, files) in SOURCES.items():
    digest, size = files['qrels']
    qrels_file = Resource(
        f'nano-beir-{v1_id}-rcp-qrels.parquet',
        sources=[f'https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/{subset}/qrels.parquet'],
        hash=digest, size=size)
    qrels = RcpQrels(
        f'nano-beir-{v1_id}-rcp-qrels', source=qrels_file)

    base = nano_benchmarks[f'nano-beir-{v1_id}']
    name = f'nano-beir-{v1_id}-rcp'
    rcp_benchmarks[name] = Benchmark(
        name, docs=base.docs, queries=base.queries, qrels=qrels, citation=CITATION)


# Registration
# -----------------------------------------
irds.register(*rcp_benchmarks.values())
irds.register(Suite('nano-beir-rcp', benchmarks=list(rcp_benchmarks.values()),
                    citation=CITATION))
