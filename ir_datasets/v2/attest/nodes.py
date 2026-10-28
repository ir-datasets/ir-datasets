"""Node classes for attestations, and the declarations that make up the ``attest`` vocabulary."""
from ..base import Node

from . import records, vocabulary
from .provider import attest

CLASSES = attest.declare_graph_classes(vocabulary)

#: dataset node -> attestation about it. Informational: nothing is built from an attestation.
ATTESTATION = attest.declare_graph_property(
    'attestation', structural=False, range_type=CLASSES[vocabulary.Attestation],
    desc='a node -> an attestation someone made about it')
RECORD = attest.declare_graph_property('record', range='json', domain=CLASSES[vocabulary.Attestation],
                                       desc='the attestation\'s content, as a JSON document')
SIGNATURE = attest.declare_graph_property('signature', range='literal', domain=CLASSES[vocabulary.Attestation],
                                          desc='an Ed25519 signature over the record, by the signed_by identity')
SIGNED_BY = attest.declare_graph_property('signedBy', structural=False, domain=CLASSES[vocabulary.Attestation],
                                          range_type=CLASSES[vocabulary.IdentityAttestation],
                                          desc='attestation -> the identity that signed it')
VERIFIED = attest.declare_graph_property('verified', range='literal', domain=CLASSES[vocabulary.Attestation],
                                         desc='true if the server checked the signature when it received the record')

_TYPE_OF_KIND = {
    records.HASH: CLASSES[vocabulary.HashAttestation],
    records.VISUAL: CLASSES[vocabulary.VisualAttestation],
    records.EVALUATION: CLASSES[vocabulary.EvaluationAttestation],
    records.IDENTITY: CLASSES[vocabulary.IdentityAttestation],
}


def type_of_kind(kind):
    """The graph class of an attestation kind (``'hash'`` -> ``attest:HashAttestation``)."""
    return _TYPE_OF_KIND[kind]


class Attestation(Node):
    """One attestation, resolved from a server record. Never constructed from ``name`` alone;
    use ``from_row``."""
    type = CLASSES[vocabulary.Attestation]

    def __init__(self, id, *, kind, subject, record, signature=None, signed_by=None,
                 verified=False, received_at=None):
        super().__init__(id)
        self.kind, self.subject, self.record = kind, subject, record
        self.signature, self.signed_by, self.verified = signature, signed_by, verified
        self.received_at = received_at
        self.type = _TYPE_OF_KIND[kind]

    @classmethod
    def from_row(cls, row):
        return cls(row['id'], kind=row['type'], subject=row['subject'], record=row['record'],
                   signature=row.get('signature'), signed_by=row.get('signed_by'),
                   verified=row.get('verified', False), received_at=row.get('received_at'))

    @property
    def signed(self):
        return bool(self.signature)

    def attestation_edges(self):
        """The edge from the attested node to this attestation, and from it to its signer."""
        out = []
        if self.subject:
            out.append((self.subject, ATTESTATION, f'attest:{self.name}'))
        if self.signed_by and self.signed_by != self.name:
            out.append((f'attest:{self.name}', SIGNED_BY, f'attest:{self.signed_by}'))
        return out
