"""Format node classes: the node type *is* the format.

    docs   = TsvDocs('antique-docs', source=docs_file, lang='en')
    qrels  = TrecQrels('antique-test-qrels', source=qrels_file, defs=QREL_DEFS)

Each class subclasses its entity base (``DocTable``, ``QueryTable``, ``QrelTable``, ...), so
it inherits everything entity-shaped -- the default/legacy record API, docstore
handling, lookup -- and adds only how bytes become records. This mirrors v1,
where ``ir_datasets.formats.TsvDocs`` is itself a docs handler.

Format arguments (``cls``, ``skip_first_line``, ...) and node arguments
(``lang``, ``desc``, ``defs``, ...) share one flat keyword list, which is what an
author wants; they do not collide.

The entity bases still accept ``parser=`` for cases where the format is chosen at
runtime rather than written down -- a wrapper inferring columns from a remote
dataset, say. These classes are the common path; that is the escape hatch.
"""
from ir_datasets import formats as _v1
from ir_datasets.formats import GenericDoc, GenericDocPair, GenericQuery, TrecQuery
from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, PickleLz4FullStore

from .nodes import DocPairTable, DocTable, QrelTable, QueryTable, RunTable


class _V1TsvDocs(_v1.TsvDocs):
    """v1 TsvDocs, but with the docstore where the node wants it.

    v1 puts the index next to the source file (``<source>.pklz4``). With
    content-addressed sources that would be an opaque ``downloads/<md5>.pklz4``,
    so the node supplies a readable, format-versioned path instead.
    """

    def __init__(self, *args, store_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._store_path = store_path

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=str(self._store_path),
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=self._doc_store_index_fields or ['doc_id'],
            size_hint=self._docstore_size_hint,
            count_hint=self._count_hint,
            options=options,
        )


class Parser:
    """Builds the v1 handler for a node.

    An implementation detail of the classes below, and the thing ``parser=``
    takes when a format must be chosen at runtime.
    """
    name = None

    def build(self, source, node):
        raise NotImplementedError

    def __repr__(self):
        return self.name or type(self).__name__


class _TsvDocsParser(Parser):
    name = 'TsvDocs'

    def __init__(self, cls, skip_first_line, index_fields):
        self.cls = cls
        self.skip_first_line = skip_first_line
        self.index_fields = index_fields

    def build(self, source, node):
        return _V1TsvDocs(
            source,
            doc_cls=self.cls,
            doc_store_index_fields=self.index_fields,
            lang=node.lang,
            skip_first_line=self.skip_first_line,
            docstore_size_hint=node.docstore_size_hint,
            count_hint=node.count_hint,
            store_path=node.docstore_path,
        )


class _TsvQueriesParser(Parser):
    name = 'TsvQueries'

    def __init__(self, cls):
        self.cls = cls

    def build(self, source, node):
        return _v1.TsvQueries(source, query_cls=self.cls, lang=node.lang)


class _TrecQrelsParser(Parser):
    name = 'TrecQrels'

    def __init__(self, format_3col):
        self.format_3col = format_3col

    def build(self, source, node):
        return _v1.TrecQrels(source, node.defs or {},
                             format_3col=self.format_3col)


class _TrecScoredDocsParser(Parser):
    name = 'TrecScoredDocs'

    def __init__(self, negate_score):
        self.negate_score = negate_score

    def build(self, source, node):
        return _v1.TrecScoredDocs(source, negate_score=self.negate_score)


class _TsvDocPairsParser(Parser):
    name = 'TsvDocPairs'

    def __init__(self, cls):
        self.cls = cls

    def build(self, source, node):
        return _v1.TsvDocPairs(source, docpair_cls=self.cls)


class _TrecDocsParser(Parser):
    name = 'TrecDocs'

    def __init__(self, encoding, path_globs, parser, expected_file_count):
        self.encoding = encoding
        self.path_globs = path_globs
        self.parser = parser
        self.expected_file_count = expected_file_count

    def build(self, source, node):
        # v1's TrecDocs already handles both a plain file and a tar (globbing
        # members in streaming mode) off the same source object -- no special
        # pipeline step needed for path_globs, unlike TarExtract's one-member
        # case (.member()).
        return _v1.TrecDocs(
            source,
            encoding=self.encoding,
            path_globs=self.path_globs,
            parser=self.parser,
            expected_file_count=self.expected_file_count,
            lang=node.lang,
            docstore_size_hint=node.docstore_size_hint,
            count_hint=node.count_hint,
            docstore_path=node.docstore_path,
        )


class _TrecQueriesParser(Parser):
    name = 'TrecQueries'

    def __init__(self, qtype, qtype_map, encoding, remove_tags):
        self.qtype = qtype
        self.qtype_map = qtype_map
        self.encoding = encoding
        self.remove_tags = remove_tags

    def build(self, source, node):
        return _v1.TrecQueries(
            source, qtype=self.qtype, qtype_map=self.qtype_map,
            encoding=self.encoding, lang=node.lang,
            remove_tags=self.remove_tags)


# ── Public format nodes ──────────────────────────────────────────────────────

class TsvDocs(DocTable):
    """``doc_id<TAB>text`` per line."""

    def __init__(self, name, *, cls=GenericDoc, skip_first_line=False,
                 index_fields=None, **kwargs):
        super().__init__(
            name,
            parser=_TsvDocsParser(cls, skip_first_line, index_fields),
            **kwargs)


class TsvQueries(QueryTable):
    """``query_id<TAB>text`` per line."""

    def __init__(self, name, *, cls=GenericQuery, **kwargs):
        super().__init__(name, parser=_TsvQueriesParser(cls), **kwargs)


class TrecQrels(QrelTable):
    """``query_id iteration doc_id relevance`` (or 3-column)."""

    def __init__(self, name, *, format_3col=False, **kwargs):
        super().__init__(name, parser=_TrecQrelsParser(format_3col), **kwargs)


class TrecScoredDocs(RunTable):
    """``query_id iteration doc_id rank score [runtag]``."""

    def __init__(self, name, *, negate_score=False, **kwargs):
        super().__init__(name, parser=_TrecScoredDocsParser(negate_score),
                         **kwargs)


class TsvDocPairs(DocPairTable):
    """``query_id doc_id doc_id`` per line."""

    def __init__(self, name, *, cls=GenericDocPair, **kwargs):
        super().__init__(name, parser=_TsvDocPairsParser(cls), **kwargs)


class TrecDocs(DocTable):
    """TREC SGML-ish docs (``<DOC><DOCNO>...</DOCNO>...</DOC>``), from a plain
    file or (with ``path_globs``) globbed out of a tar archive."""

    def __init__(self, name, *, encoding=None, path_globs=None,
                 parser='BS4', expected_file_count=None, **kwargs):
        super().__init__(
            name,
            parser=_TrecDocsParser(encoding, path_globs, parser, expected_file_count),
            **kwargs)


class TrecQueries(QueryTable):
    """TREC SGML-ish queries (``<top><num>...<title>...</top>``)."""

    def __init__(self, name, *, qtype=TrecQuery, qtype_map=None, encoding=None,
                 remove_tags=('</title>',), **kwargs):
        super().__init__(
            name,
            parser=_TrecQueriesParser(qtype, qtype_map, encoding, remove_tags),
            **kwargs)
