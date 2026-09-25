import json
from typing import NamedTuple

import ir_datasets
from ir_datasets.datasets.base import Dataset, YamlDocumentation
from ir_datasets.formats import BaseDocs, BaseQrels, BaseQueries, GenericDoc, GenericQuery, TrecQrel
from ir_datasets.indices import PickleLz4FullStore, DEFAULT_DOCSTORE_OPTIONS
from ir_datasets.util import count_hint, use_docstore

_logger = ir_datasets.log.easy()


NAME = 'core'

QRELS_DEFS = {
    1: 'relevant: judged relevant by NIST assessors for the TREC 2023 Deep Learning track (10 per query)',
    0: 'distractor: not relevant, but ranked highly by a Reciprocal Rank Fusion of top TREC 2023 Deep Learning track '
       'runs (100 per query)',
}

QRELS_TYPES = {'relevant': 1, 'distractor': 0}


class CoreDocument(NamedTuple):
    doc_id: str
    title: str
    headings: str
    text: str
    def default_text(self):
        """
        title + headings + text
        """
        return f'{self.title} {self.headings} {self.text}'


# The corpus splits of each track, smallest first: (name, number of shard files). Each split holds only the
# documents it adds on top of the previous ones -- the corpus of a given size is the concatenation of every split
# up to and including it, so docs_iter() of a larger corpus always starts with the documents of every smaller one.
SPLITS = {
    'passage': [('core', 1), ('10k', 1), ('100k', 1), ('1M', 1), ('10M', 9), ('100M', 90)],
    'document': [('core', 1), ('10k', 1), ('100k', 1), ('1M', 1), ('10M', 9)],
}

DOC_TYPES = {
    'passage': GenericDoc,
    'document': CoreDocument,
}


class CoreDocs(BaseDocs):
    def __init__(self, dsid, track, dlcs):
        super().__init__()
        self._dsid = dsid
        self._track = track
        self._dlcs = dlcs
        self._doc_cls = DOC_TYPES[track]

    @use_docstore
    def docs_iter(self):
        for dlc in self._dlcs:
            with dlc.stream() as stream:
                for line in stream:
                    data = json.loads(line)
                    if self._doc_cls is GenericDoc:
                        yield GenericDoc(str(data['_id']), data['text'])
                    else:
                        yield CoreDocument(str(data['_id']), data['title'], data['headings'], data['text'])

    def docs_cls(self):
        return self._doc_cls

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path()}/{self._dsid}/docs.pklz4',
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
            count_hint=count_hint(self._dsid),
            options=options,
        )

    def docs_count(self):
        if self.docs_store().built():
            return self.docs_store().count()
        # otherwise falls back onto the count in metadata.json

    def docs_namespace(self):
        # Shared by every corpus size of a track: they hold the same documents under the same ids.
        return f'{NAME}/{self._track}'

    def docs_lang(self):
        return 'en'


class CoreQrels(BaseQrels):
    def __init__(self, dlc):
        self._dlc = dlc

    def qrels_path(self):
        return self._dlc.path()

    def qrels_iter(self):
        with self._dlc.stream() as stream:
            for line in stream:
                data = json.loads(line)
                # The source files store query ids as JSON numbers; ir_datasets ids are always strings.
                yield TrecQrel(str(data['query-id']), str(data['corpus-id']), QRELS_TYPES[data['type']], '0')

    def qrels_cls(self):
        return TrecQrel

    def qrels_defs(self):
        return QRELS_DEFS


class CoreQueries(BaseQueries):
    """
    The passage and document tracks share one query file, but each only has judgments for a subset of its
    queries -- only those are part of the track.
    """
    def __init__(self, dlc, qrels, track):
        super().__init__()
        self._dlc = dlc
        self._qrels = qrels
        self._track = track

    def queries_iter(self):
        query_ids = {qrel.query_id for qrel in self._qrels.qrels_iter()}
        with self._dlc.stream() as stream:
            for line in stream:
                data = json.loads(line)
                query_id = str(data['_id'])
                if query_id in query_ids:
                    yield GenericQuery(query_id, data['text'])

    def queries_cls(self):
        return GenericQuery

    def queries_namespace(self):
        # The query ids are those of the TREC 2023 Deep Learning track.
        return 'msmarco'

    def queries_lang(self):
        return 'en'


def _init():
    base_path = ir_datasets.util.home_path() / NAME
    dlc = ir_datasets.util.DownloadConfig.context(NAME, base_path)
    documentation = YamlDocumentation(f'docs/{NAME}.yaml')

    base = Dataset(documentation('_'))

    subsets = {}
    for track, splits in SPLITS.items():
        qrels = CoreQrels(dlc[f'{track}/qrels'])
        queries = CoreQueries(dlc['queries'], qrels, track)
        # The largest corpus of a track is the track itself (e.g. core/passage is the 100M-passage corpus); every
        # smaller one is a subset named after its size (e.g. core/passage/10k), down to core/<track>/core: only
        # the relevant documents and distractors.
        docs_dlcs = []
        for i, (size, num_shards) in enumerate(splits):
            if num_shards == 1:
                docs_dlcs.append(dlc[f'{track}/docs/{size}'])
            else:
                docs_dlcs.extend(dlc[f'{track}/docs/{size}/{shard}'] for shard in range(num_shards))
            dsid = track if i == len(splits) - 1 else f'{track}/{size}'
            subsets[dsid] = Dataset(
                CoreDocs(f'{NAME}/{dsid}', track, list(docs_dlcs)),
                queries,
                qrels,
                documentation(dsid),
            )

    ir_datasets.registry.register(NAME, base)
    for s in sorted(subsets):
        ir_datasets.registry.register(f'{NAME}/{s}', subsets[s])

    return base, subsets


base, subsets = _init()
