"""Touché Image Search -- a v2 dataset family (one corpus: ~24K web images
for the argument-retrieval-with-image-support task).

No benchmark of its own -- it's judged only by Touché 2022 Task 3, defined
in ``touche.py`` (imported by reference from here, same cross-file pattern
as ``disks45.py``/``trec_adhoc.py``).

Reuses v1's ``ToucheImageDocs`` directly as a v2 ``parser=`` wrapper (three
zip archives -- main captions/metadata, node/graph info, and PNG images --
opened together in lockstep), same "thin layer over v1 machinery" pattern as
every other migrated family.
"""
from ir_datasets.formats import ToucheImageDocs as _V1ToucheImageDocs
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser


class _ToucheImageDocsParser(Parser):
    name = 'ToucheImageDocs'

    def build(self, source, node):
        main, nodes, png = source
        return _V1ToucheImageDocs(main, nodes, png, language=node.lang, count_hint=node.count_hint)


# Files
# -----------------------------------------
images_main_file = Resource('touche-image-2022-06-13-main.zip',
    sources=['https://zenodo.org/record/6873575/files/touche22-image-search-main.zip'],
    md5='e59b1c724d976af27596b5c8ad310fd5',
    size=4_498_749_006,
)
images_nodes_file = Resource('touche-image-2022-06-13-nodes.zip',
    sources=['https://zenodo.org/record/6873575/files/touche22-image-search-nodes.zip'],
    md5='97b7117d02668fa3e93095d277efd56b',
    size=5_424_503_960,
)
images_png_file = Resource('touche-image-2022-06-13-png.zip',
    sources=['https://zenodo.org/record/6873575/files/touche22-image-search-png-images.zip'],
    md5='e2965b221248ba23a288135f757efae1',
    size=17_851_724_760,
)

# Tables
# -----------------------------------------
docs = DocTable('touche-image-2022-06-13-docs',
    source=[images_main_file, images_nodes_file, images_png_file],
    parser=_ToucheImageDocsParser(),
    lang='en',
    count_hint=23_841,
)


# Registration
# -----------------------------------------
irds.register(docs)
