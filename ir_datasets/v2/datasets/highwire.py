"""Highwire -- a v2 dataset family (corpus for TREC Genomics track
2006/2007; the benchmarks themselves live in ``trec_genomics.py``, which
also covers medline.py's 2004/2005 years of the same track -- see that
module's docstring).

Reuses v1's ``HighwireDocs`` handler class directly as a v2 ``parser=``
wrapper, same "thin layer over v1 machinery" pattern as
``clinicaltrials.py``/``car.py``.

The corpus is split across 58 per-journal zip Resources (one per entry in
v1's ``SOURCES``) plus a single "legalspans" Resource giving byte
offsets/lengths of the extractable passage spans within each doc -- v1's
``HighwireDocs`` walks the two in lockstep (one legalspans line-group per
doc, in the same order the zip files are opened), so both must reach the
parser together. Since ``Table.__init__`` only accepts one ``source=``
value, all 59 Resources are passed as a single list (structural edges are
just "every Resource this source expression bottoms out in" -- see
``nodes.source_resources``, which walks lists but not dicts) and the parser
splits the flat list back into ``(dlcs_by_source_name, legalspans)`` before
handing it to v1.

"""
from ir_datasets.datasets.highwire import HighwireDocs as _V1HighwireDocs
from ir_datasets.datasets.highwire import SOURCES
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser


class _HighwireDocsParser(Parser):
    name = 'HighwireDocs'

    def build(self, source, node):
        *corpus_sources, legalspans_source = source
        dlcs = dict(zip(SOURCES, corpus_sources))
        return _V1HighwireDocs(dlcs, legalspans_source)


_CORPUS_SOURCES = {
    'ajepidem': ('d7db27233b28245724f3212a5b3cd659', 25_454_608),
    'ajpcell': ('5d5cdc8e6cdaeeb924bc5af603e4478e', 64_758_393),
    'ajpendometa': ('2ce8f78e813e60e424136f010f9e0af0', 50_030_877),
    'ajpgastro': ('0fe9e5091ac4f3adcc2325ed6d881964', 50_853_597),
    'ajpheart': ('4152b0904cc66e95f953a5c097d7ad77', 104_053_949),
    'ajplung': ('ca5a42357e4d8413ac32ffc21cc7a25f', 50_558_558),
    'ajprenal': ('6c21485c1858eda002766ec364af576a', 41_011_735),
    'alcohol': ('ca34997e36b919e64b99e12606c511cc', 10_152_433),
    'andrology': ('d715933ff1c96689e9647bbecdaaf599', 7_454_866),
    'annonc': ('32234a3d00ea71b923006dd1f90c2dd5', 16_794_376),
    'bjanast': ('5a9c47c19d6bd33b8e30c27be9e41d25', 21_847_467),
    'bjp': ('93707a10817b15773e67bf1a2d3b2e24', 17_371_510),
    'blood': ('2216234fc6d66445026cc38c9c714855', 219_220_876),
    'carcinogenesis': ('7f4d797e2b779c9490eb9e523ebf628f', 38_205_239),
    'cercor': ('60cfd9eebf42e1602cd69805de409e04', 23_218_757),
    'development': ('f71ca283a55848a51a53c73a5af7db3c', 64_783_714),
    'diabetes': ('e6fcc28d779833b5b07fc8f2d2b4077e', 39_020_405),
    'endocrinology': ('85cf5f2870ccf333114e59ce1ab4d1db', 109_222_542),
    'euroheartj': ('099dd344c7a98e3b6d3dbed2cdb17fb7', 15_462_924),
    'glycobiology': ('4e390a903cbf2381c2eb530a3e7b3b68', 15_210_749),
    'humanrep': ('b30e1d36c8c551f33c0a8cf4cc09dfce', 52_745_668),
    'humolgen': ('81ae00345d24ec20a542e9c151181c93', 61_054_993),
    'ijepidem': ('d9e1ea6aaabcc04fb57f26a756a49b40', 14_035_365),
    'intimm': ('17ebda5809cd22a23ac92310a103fb9c', 23_787_976),
    'jantichemo': ('e8b6088e655078ff40647af9a3cbd30f', 30_906_187),
    'jappliedphysio': ('7a5406c65b0385a5cc493fede1424b27', 109_740_124),
    'jbc-1995': ('690f25229a7040627d32be56bdcf4556', 77_817_763),
    'jbc-1996': ('646a254c22f40861966fccacea0b1b5b', 34_630_282),
    'jbc-1997': ('5dfb2a85548e3da5305bd5d7e051c2fa', 62_855_350),
    'jbc-1998': ('43cc8a9797700d99d17cd14596a08028', 61_361_036),
    'jbc-1999': ('d5972b628653f5719a7ca3d04d238c10', 51_850_260),
    'jbc-2000': ('d00f5eb17beb680752fa1711c6fe2cb8', 116_888_541),
    'jbc-2001': ('a8e8e08a530754a5c2de9197c1d4259c', 72_164_976),
    'jbc-2002': ('93e8ae6565afc20affcf785fb6a34646', 125_104_356),
    'jbc-2003': ('7ddf90fede68fc0a94a674557e457063', 80_012_227),
    'jbc-2004': ('d50a6264161f1aed2545737ace31ad32', 138_441_109),
    'jbc-2005': ('4bac0f72d3845b6266a5bdf448d215b4', 113_908_404),
    'jcb': ('dba03aa0b7c8e1b72817afc880f5ff6c', 98_000_404),
    'jclinicalendometa': ('fdc5a8b41be0b58beabc822dc28660e2', 7_262_318),
    'jcs': ('cd506b2d653c41916450f7f46a78d402', 57_046_291),
    'jexpbio': ('2072dad39b3fcdc58cb5a522c35f1b59', 43_249_960),
    'jexpmed': ('6a6b6fd5d883c99be91d84e8663b6293', 73_328_992),
    'jgenphysio': ('4cb4bbb2c979e992e2668c0a6585d9b4', 25_699_101),
    'jgenviro': ('ba0fa54e2b8e7101c2c5008cb89f4c47', 42_041_939),
    'jhistocyto': ('def37c9b86e4ddbffd6a845c642ef009', 25_544_265),
    'jnci': ('649621e050ab0f0e73eb626002cf2a91', 36_174_794),
    'jneuro': ('cfd69effb5d2e0eb845173095e21888c', 71_785_901),
    'mcp': ('198e85d2c65d3c3964195e955660409d', 9_910_430),
    'microbio': ('06840bddca76f1c7d8644ca989189368', 48_411_405),
    'molbiolevol': ('8b1eb9eceaea7da237f177d235a058a5', 26_619_856),
    'molendo': ('a2f55efa0e3bffe5d736316585540b62', 38_090_065),
    'molhumanrep': ('125a49b8f11ad1a78eda3ce03f867ef4', 14_601_617),
    'nar': ('63c34a05440cd0259b06aef3edb22ea3', 132_453_207),
    'nephrodiatransp': ('914e51d8ac94e85d2bbdc8afb9206731', 39_541_854),
    'peds': ('db8266c186e902715beb511f12bcd16a', 15_511_622),
    'physiogenomics': ('eeda5cc0b14ad36402cdc6ffd964e3c1', 13_767_348),
    'rheumatolgy': ('3e840dfb598dbefe630cdb1b412f3c57', 22_015_632),
    'rna': ('f48dd7fe09ae06609d6a8dc5e77aae2f', 11_853_002),
    'toxsci': ('0093349b4f41e5431b1d832d13738010', 34_838_564),
}
assert list(_CORPUS_SOURCES) == SOURCES


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    corpus_files = [
        Resource(f'highwire-{source}.zip',
            sources=[f'https://dmice.ohsu.edu/trec-gen/data/2006/documents/{source}.zip'],
            hash=f'md5:{md5}', size=size)
        for source, (md5, size) in _CORPUS_SOURCES.items()
    ]
    legalspans_file = Resource('highwire-legalspans.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2006/documents/legalspans.txt'],
        hash='md5:24c7bbed8eb3bdd2daf3f3ab2c1963b2',
        size=236_522_265,
    )

    # Tables
    # -----------------------------------------
    docs = DocTable('highwire',
        source=[*corpus_files, legalspans_file],
        parser=_HighwireDocsParser(),
        count_hint=162_259,
    )


# Registration
# -----------------------------------------
irds.register(docs)
