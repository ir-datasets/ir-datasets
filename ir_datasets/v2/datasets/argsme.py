"""args.me -- a v2 dataset family (argument search / computational
argumentation corpora, used by the Touché shared tasks).

Reuses v1's ``ArgsMeDocs``/``ArgsMeProcessedDocs``/``ArgsMeCombinedDocs``
handler classes directly as v2 ``parser=`` wrappers, same "thin layer over
v1 machinery" pattern as ``clinicaltrials.py``/``car.py``/``highwire.py``.
All three only ever call ``.stream()`` (``ArgsMeProcessedDocs`` also wraps
the stream in ``io.TextIOWrapper``) and, for the docstore's default naming,
``.path()`` -- both of which a v2 ``Readable`` pipeline provides, so no
subclassing is needed for the plain and processed variants.

Each Zenodo download is a single zip containing one JSON file
(``.zip_member(...)``, no ``.cache()`` needed -- the zip itself is already
the one thing fetched/cached, same reasoning as ``wikir.py``); the one
processed-subset download is a tar.gz containing a CSV, read through
``TextIOWrapper``, so it needs ``.cache(path)`` (same reasoning as
``cranfield.py``/``msmarco_passage.py``).

``argsme-2020-04-01-docs`` is a *derived* table: v1's ``ArgsMeCombinedDocs``
concatenates the docs of five of the plain subsets above (debateorg,
debatepedia, debatewise, idebate, parliamentary). Rather than reference
those tables' already-built handlers, the combined table's ``source=`` is
just the list of their five underlying Resources again (structural edges
are "every Resource a source expression bottoms out in" -- see
``nodes.source_resources``), and the parser re-wraps each one in its own
``ArgsMeDocs`` before handing the list to ``ArgsMeCombinedDocs`` -- same flat
list + split-and-rebuild shape ``highwire.py`` uses for its 58 corpus zips.
``ArgsMeCombinedDocs`` needs an on-disk ``path`` for its docstore (it has no
Resource of its own to derive one from); ``_ArgsMeCombinedDocs`` below
points that at ``node.docstore_path`` instead, same fix ``car.py`` applies
for the same reason.

NOTE: the following datasets are defined in the (not yet migrated)
``touche.py`` instead, and are not this file's concern:
 - argsme/1.0/touche-2020-task-1/uncorrected
 - argsme/2020-04-01/touche-2020-task-1
 - argsme/2020-04-01/touche-2020-task-1/uncorrected
 - argsme/2020-04-01/touche-2021-task-1
 - argsme/2020-04-01/processed/touche-2022-task-1
"""
from ir_datasets.formats import ArgsMeCombinedDocs as _V1ArgsMeCombinedDocs
from ir_datasets.formats import ArgsMeDocs as _V1ArgsMeDocs
from ir_datasets.formats import ArgsMeProcessedDocs as _V1ArgsMeProcessedDocs
from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, PickleLz4FullStore
from ir_datasets.util import home_path
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser

BASE = home_path() / 'argsme'


class _ArgsMeDocsParser(Parser):
    name = 'ArgsMeDocs'

    def build(self, source, node):
        return _V1ArgsMeDocs(source, language=node.lang, count_hint=node.count_hint)


class _ArgsMeProcessedDocsParser(Parser):
    name = 'ArgsMeProcessedDocs'

    def build(self, source, node):
        return _V1ArgsMeProcessedDocs(source, language=node.lang, count_hint=node.count_hint)


class _ArgsMeCombinedDocs(_V1ArgsMeCombinedDocs):
    """v1 ArgsMeCombinedDocs, but with the docstore where the node wants it
    (its v1 ``path`` argument is otherwise just a made-up on-disk name for a
    derived collection with no Resource of its own -- see the module
    docstring, and ``car.py`` for the same fix applied to a real path
    collision)."""

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=str(self._path),
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
            count_hint=self._count_hint,
            options=options,
        )


class _ArgsMeCombinedDocsParser(Parser):
    name = 'ArgsMeCombinedDocs'

    def __init__(self, count_hints):
        self.count_hints = count_hints

    def build(self, source, node):
        sub_docs = [
            _V1ArgsMeDocs(sub_source, language=node.lang, count_hint=count_hint)
            for sub_source, count_hint in zip(source, self.count_hints)
        ]
        return _ArgsMeCombinedDocs(node.docstore_path, sub_docs,
            language=node.lang, count_hint=node.count_hint)


CITATION = 'dblp:conf/argmining/WachsmuthPKAPQD17; dblp:conf/ki/AjjourWKPHS19'

with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    docs_1_0_file = Resource('argsme-1.0-docs.zip',
        sources=['https://zenodo.org/record/3274636/files/argsme.zip'],
        md5='c2512648f46a403f8e5e1dc96779e357',
        size=238_078_064,
    ).zip_member('args-me.json')
    docs_1_0_cleaned_file = Resource('argsme-1.0-cleaned-docs.zip',
        sources=['https://zenodo.org/record/4139439/files/argsme-1.0-cleaned.zip'],
        md5='fb0837103a4860e1d4536174f55b12c3',
        size=236_787_622,
    ).zip_member('args-me-1.0-cleaned.json')

    debateorg_file = Resource('argsme-2020-04-01-debateorg-docs.zip',
        sources=['https://zenodo.org/record/3734893/files/debateorg.zip'],
        md5='0368ee47ce0ec8bed837c7e22c024493',
        size=1_150_846_141,
    ).zip_member('debateorg.json')
    debatepedia_file = Resource('argsme-2020-04-01-debatepedia-docs.zip',
        sources=['https://zenodo.org/record/3734893/files/debatepedia.zip'],
        md5='bde8e3ed832c19ca5ed8ed1506a862e8',
        size=184_347_726,
    ).zip_member('debatepedia.json')
    debatewise_file = Resource('argsme-2020-04-01-debatewise-docs.zip',
        sources=['https://zenodo.org/record/3734893/files/debatewise.zip'],
        md5='5e5c498a5f657ed7d02e06016e9ce3b1',
        size=77_388_912,
    ).zip_member('debatewise.json')
    idebate_file = Resource('argsme-2020-04-01-idebate-docs.zip',
        sources=['https://zenodo.org/record/3734893/files/idebate.zip'],
        md5='5b888c94cce740f1216c063e5e47c74c',
        size=20_241_730,
    ).zip_member('idebate.json')
    parliamentary_file = Resource('argsme-2020-04-01-parliamentary-docs.zip',
        sources=['https://zenodo.org/record/3734893/files/parliamentary.zip'],
        md5='c80d932c953b64fb300f13d0d93096bb',
        size=27_319,
    ).zip_member('parliamentary.json')

    processed_file = Resource('argsme-2020-04-01-processed-docs.tar.gz',
        sources=['https://zenodo.org/record/6873574/files/args_processed_04_01.tar.gz'],
        md5='43bfce957df69bf59b3d59744eb73ded',
        size=1_547_009_833,
    ).member('args_processed.csv').cache(BASE / '2020-04-01-processed.csv')

    # Tables
    # -----------------------------------------
    docs_1_0 = DocTable('argsme-1.0-docs',
        source=docs_1_0_file,
        parser=_ArgsMeDocsParser(),
        count_hint=387_692,
        citation=CITATION,
    )
    docs_1_0_cleaned = DocTable('argsme-1.0-cleaned-docs',
        source=docs_1_0_cleaned_file,
        parser=_ArgsMeDocsParser(),
        count_hint=382_545,
        citation=CITATION,
    )

    docs_2020_04_01_debateorg = DocTable('argsme-2020-04-01-debateorg-docs',
        source=debateorg_file,
        parser=_ArgsMeDocsParser(),
        count_hint=338_620,
        citation=CITATION,
    )
    docs_2020_04_01_debatepedia = DocTable('argsme-2020-04-01-debatepedia-docs',
        source=debatepedia_file,
        parser=_ArgsMeDocsParser(),
        count_hint=21_197,
        citation=CITATION,
    )
    docs_2020_04_01_debatewise = DocTable('argsme-2020-04-01-debatewise-docs',
        source=debatewise_file,
        parser=_ArgsMeDocsParser(),
        count_hint=14_353,
        citation=CITATION,
    )
    docs_2020_04_01_idebate = DocTable('argsme-2020-04-01-idebate-docs',
        source=idebate_file,
        parser=_ArgsMeDocsParser(),
        count_hint=13_522,
        citation=CITATION,
    )
    docs_2020_04_01_parliamentary = DocTable('argsme-2020-04-01-parliamentary-docs',
        source=parliamentary_file,
        parser=_ArgsMeDocsParser(),
        count_hint=48,
        citation=CITATION,
    )

    docs_2020_04_01_processed = DocTable('argsme-2020-04-01-processed-docs',
        source=processed_file,
        parser=_ArgsMeProcessedDocsParser(),
        count_hint=365_408,
        citation=CITATION,
    )

    docs_2020_04_01 = DocTable('argsme-2020-04-01-docs',
        source=[debateorg_file, debatepedia_file, debatewise_file, idebate_file, parliamentary_file],
        parser=_ArgsMeCombinedDocsParser([338_620, 21_197, 14_353, 13_522, 48]),
        count_hint=387_740,
        citation=CITATION,
    )


# Registration
# -----------------------------------------
irds.register(
    docs_1_0, docs_1_0_cleaned,
    docs_2020_04_01_debateorg, docs_2020_04_01_debatepedia, docs_2020_04_01_debatewise,
    docs_2020_04_01_idebate, docs_2020_04_01_parliamentary,
    docs_2020_04_01_processed, docs_2020_04_01,
)
