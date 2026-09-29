"""C4 -- a v2 dataset family (the "colossal, cleaned Common Crawl corpus"),
covering only what v1's own module builds: the ``en-noclean`` (train) split,
plus the TREC Misinfo 2021 benchmark that judges it.

Unlike every other migrated family, the corpus here is **not** a fixed list
of Resources declared up front. v1's ``C4Docs`` fetches one small manifest
JSON (``en-noclean/sources``, ~240KB gzipped) that lists ~7168 shard URLs
plus each shard's md5/size/doc_count/checkpoint_freq, and only *then*
constructs one v1 ``Download`` per shard, lazily, the first time
``docs_iter()``/``docs_count()``/etc. actually needs it (see
``_docs_sources()`` in ``ir_datasets/datasets/c4.py``). There is no way to
know the shard list -- let alone declare ~7168 individual ``Resource``
nodes -- without fetching and reading that manifest, and doing so at import
or freeze time is exactly the kind of eager, large-scale work v2 nodes are
meant to avoid. So, following the same "thin reuse, don't reimplement"
precedent as ``gov.py``, this module keeps v1's ``C4Docs``/``C4Source``/
``C4SourceIter``/``C4Docstore`` entirely as-is and simply constructs
``C4Docs`` the same way v1's ``_init()`` does, passing it the two v2
``Resource``s for the manifest and the checkpoints archive directly (v2
``Resource``/pipeline objects satisfy the same ``.path()``/``.stream()``
surface v1's own ``dlc`` objects do -- confirmed by ``clinicaltrials.py``/
``highwire.py`` passing v2 Resources straight into v1 handler
constructors). ``C4Docs`` does its own per-shard caching (under
``<home>/c4/en.noclean/...``, keyed by each shard's own md5) regardless of
where a v2 node's ``docstore_path`` would otherwise point, so -- again
like ``gov.py``/``clinicaltrials.py`` reusing v1's own path management --
``base_path`` here is simply v1's own ``ir_datasets.util.home_path()/'c4'``,
not derived from the node.

Doc count: v1 never hardcodes one either (``docs_count()`` sums each
shard's ``doc_count`` from the manifest, once fetched) -- ``count_hint`` below
is instead taken from ``test/integration/c4.py``'s integration-test count for
``c4/en-noclean-tr`` (1,063,805,381), the only authoritative number
available without downloading the manifest ourselves.

The TREC Misinfo 2021 queries (``trec-misinfo-2021/queries`` in v1's
``downloads.json``) were a manual/DUA-style download in v1 (an Active
Participants URL that now returns 401); NIST now publishes the topics file
openly at ``https://trec.nist.gov/data/misinfo/misinfo-2021-topics.xml``, so
it is an ordinary URL Resource here.
"""
import ir_datasets
from ir_datasets.datasets.c4 import C4Docs as _V1C4Docs, MisinfoQuery
from ir_datasets.util import TarExtractAll
from ir_datasets.v2 import Benchmark, DocTable, QueryTable, Resource, irds
from ir_datasets.v2.formats import Parser

BASE_PATH = ir_datasets.util.home_path() / 'c4'

misinfo_map = {'number': 'query_id', 'query': 'text', 'description': 'description',
               'narrative': 'narrative', 'disclaimer': 'disclaimer', 'stance': 'stance',
               'evidence': 'evidence'}


class _C4DocsParser(Parser):
    """Wraps v1's ``C4Docs`` directly -- see the module docstring for why the
    ~7168-shard manifest can't be turned into up-front ``Resource``
    declarations, and why ``base_path`` is v1's own well-known path rather
    than a v2 docstore path."""
    name = 'C4Docs'

    def __init__(self, source_name_filter=None, filter_name=''):
        self.source_name_filter = source_name_filter
        self.filter_name = filter_name

    def build(self, source, node):
        sources_resource, checkpoint_resource = source
        return _V1C4Docs(
            sources_resource.gunzip(),
            TarExtractAll(checkpoint_resource, BASE_PATH / 'en.noclean.checkpoints'),
            BASE_PATH,
            source_name_filter=self.source_name_filter,
            filter_name=self.filter_name,
        )


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
    en_noclean_sources_file = Resource('c4-en-noclean-sources.json.gz',
        sources=['https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/ir-datasets/c4/en.noclean.sources.json.gz'],
        md5='3faf0f3aaf3f0e5bca573e118f815991',
        size=240_518,
    )
    en_noclean_checkpoints_file = Resource('c4-en-noclean-checkpoints.tar.gz',
        sources=['https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/ir-datasets/c4/en.noclean.checkpoints.tar.gz'],
        md5='eab00c3b5202564da998466198a01298',
        size=8_983_526_491,
    )
    trec_misinfo_2021_queries_file = Resource('c4-trec-misinfo-2021-queries.xml',
        sources=['https://trec.nist.gov/data/misinfo/misinfo-2021-topics.xml'],
        md5='988ea3128eefa5814b550f5fe1a15e94',
        size=50_722,
    )

    # Tables
    # -----------------------------------------
    # Only the train split (v1 excludes validation shards via filter_name='train').
    en_noclean_tr_docs = DocTable('c4-en-noclean-tr',
        source=[en_noclean_sources_file, en_noclean_checkpoints_file],
        parser=_C4DocsParser(source_name_filter=r'en\.noclean\.c4-train', filter_name='train'),
        count_hint=1_063_805_381,
        citation='dblp:journals/jmlr/RaffelSRLNMZLL20',
    )

    trec_misinfo_2021_queries = QueryTable('c4-en-noclean-tr-trec-misinfo-2021-queries',
        source=trec_misinfo_2021_queries_file,
        parser=_TrecXmlQueriesParser(MisinfoQuery, qtype_map=misinfo_map),
        count_hint=50,
        citation='dblp:conf/trec/ClarkeMS21',
    )

    # Benchmarks
    # -----------------------------------------
    trec_misinfo_2021 = Benchmark('c4-en-noclean-tr-trec-misinfo-2021',
        docs=en_noclean_tr_docs, queries=trec_misinfo_2021_queries,
        desc='TREC Misinfo 2021 (queries only; v1 does not wire up qrels for this dataset).')


# Registration
# -----------------------------------------
irds.register(en_noclean_tr_docs, trec_misinfo_2021)
