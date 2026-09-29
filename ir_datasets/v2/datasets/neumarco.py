"""neuMARCO -- a v2 dataset family.

A machine-translated corpus (JHU HLTCOE, Sockeye 2 NMT), one per language
(Persian, Chinese, Russian), evaluated against msmarco-passage's own
*English* queries/qrels/docpairs -- not re-declared here, imported by
reference from ``msmarco_passage.py`` (see ``docs/neumarco.yaml``:
"Documents: ... translated"; "Queries: ... in English"). Every benchmark
below therefore differs from its msmarco-passage counterpart only in which
``docs`` Table it points at.

The per-language, docs-only node (v1's own bare ``neumarco/{lang}``) is just
the docs table itself, not a Benchmark wrapper: a corpus with no
queries/qrels isn't an evaluable task, same convention ``hc4.py`` uses for
its own per-language, docs-only nodes.
"""
import ir_datasets
from ir_datasets.v2 import Benchmark, Resource, TsvDocs, irds
from ir_datasets.v2.datasets.msmarco_passage import (
    CITATION, MEASURES,
    dev as passage_dev, dev_judged as passage_dev_judged, dev_small as passage_dev_small,
    train as passage_train, train_judged as passage_train_judged,
)

BASE = ir_datasets.util.home_path() / 'neumarco'

VARIANTS = {
    'train': passage_train,
    'train-judged': passage_train_judged,
    'dev': passage_dev,
    'dev-judged': passage_dev_judged,
    'dev-small': passage_dev_small,
}

with irds.defaults(lang='en'):
    main_file = Resource('neumarco-main.tar.gz',
        sources=['https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dlawrie1_jh_edu/'
                 'EQcICtPaSqFNoCZHtoeZszoB7FC362BvaPvieUSk2j30tA?download=1'],
        hash='md5:733181c211959a7c09c695bfcddaea54',
        size=3_723_728_998,
    )

    nodes = {}

    for lang3, lang2 in [('fas', 'fa'), ('zho', 'zh'), ('rus', 'ru')]:
        docs = TsvDocs(f'neumarco-{lang2}',
            source=main_file.member(f'eng-{lang3}/msmarco.collection.20210731-scale21-sockeye2-tm1.tsv')
                             .cache(BASE / f'{lang2}.tsv'),
            lang=lang2)
        nodes[f'neumarco-{lang2}'] = docs

        for suffix, parent in VARIANTS.items():
            name = f'neumarco-{lang2}-{suffix}'
            nodes[name] = Benchmark(name,
                docs=docs, queries=parent.queries, qrels=parent.qrels, docpairs=parent.docpairs,
                citation=CITATION, metrics=MEASURES,
                desc=f'neuMARCO: msmarco-passage-{suffix}, corpus translated to {lang2} '
                     f'(queries/qrels/docpairs are msmarco-passage\'s own, unchanged).')


# Registration
# -----------------------------------------
irds.register(*nodes.values())
