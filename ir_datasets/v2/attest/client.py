"""Python API for reading and submitting attestations.

    from ir_datasets.v2 import attest
    attest.attest_hash('irds:antique-docs')                    # hash the table, submit anonymously
    attest.attest_evaluation('irds:antique-test', system='Terrier BM25', measures={'nDCG@10': 0.3})
    attest.for_node('irds:antique-test')                       # what has been attested so far
"""
import os
import platform
import sys

import requests

from . import crypto, hashing, records

DEFAULT_URL = 'https://attest.ir-datasets.com'


def server_url():
    return (os.environ.get('IR_DATASETS_ATTEST_URL') or DEFAULT_URL).rstrip('/')


def environment():
    """What a record says about where it was produced (all optional, informational)."""
    from importlib import metadata

    def version(dist):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None
    env = {'python': platform.python_version(), 'os': f'{platform.system()} {platform.release()}',
           'machine': platform.machine(), 'ir_datasets': version('ir_datasets')}
    return {k: v for k, v in env.items() if v}


class AttestError(Exception):
    """The server refused a request (``status`` and its explanation)."""

    def __init__(self, status, message):
        super().__init__(f'{status}: {message}')
        self.status = status


class Client:
    def __init__(self, url=None, session=None, timeout=15):
        self.url = (url or server_url()).rstrip('/')
        self.session = session or requests.Session()
        self.timeout = timeout

    def _call(self, method, path, **kw):
        r = self.session.request(method, f'{self.url}/{path}', timeout=self.timeout, **kw)
        if r.status_code >= 400:
            try:
                message = r.json().get('error', r.text)
            except ValueError:
                message = r.text
            raise AttestError(r.status_code, message)
        return r.json()

    # -- reading ------------------------------------------------------------

    def records(self, after=0):
        """Every attestation with ``seq`` greater than ``after`` (all of them by default), in pages. A deleted one comes
        back as a tombstone, ``{"id", "seq", "deleted": True}``, so a reader that has the record can drop it."""
        while True:
            page = self._call('GET', f'?full=1&after={after}&limit=1000')['attestations']
            yield from page
            if len(page) < 1000:
                return
            after = page[-1]['seq']

    def ids(self):
        return [a['id'] for a in self._call('GET', '')['attestations'] if not a.get('deleted')]

    def get(self, *ids):
        """The records for these ids (unknown ids are left out)."""
        out = []
        for i in range(0, len(ids), 40):    # keep URLs short
            out.extend(self._call('GET', ','.join(ids[i:i + 40]))['attestations'])
        return out

    def for_node(self, node):
        return self._call('GET', node)['attestations']

    # -- writing ------------------------------------------------------------

    def register(self, username, identity=None):
        """Create an identity: generate a key pair, register its public half and save the result
        to the attest directory. Returns the ``crypto.Identity``."""
        private, public = crypto.generate_keypair()
        record = records.validate(records.IDENTITY, {'kind': records.IDENTITY, 'username': username,
                                                     'public_key': public, 'attested_at': records.now()})
        signature = crypto.sign(private, records.signing_payload(record))
        id = self._call('POST', 'register', json={'record': record, 'signature': signature})['id']
        identity = crypto.Identity(username, private, public, identity_id=id, server=self.url)
        identity.save()
        return identity

    def revoke(self, attestation, reason=None, identity=None):
        """Delete an attestation you signed (``attestation``: its id, or ``attest:<id>``). Needs the identity that
        signed it (default: the saved one): the request is signed with its key, and the server checks that signature
        against the public key of the identity that signed the attestation, which is what proves it is yours. Returns
        the id deleted. Anonymous attestations have no key to prove, so cannot be deleted."""
        identity = identity or crypto.Identity.load()
        if identity is None:
            raise ValueError('revoking needs your identity: run "ir_datasets attest register <username>" first')
        name = attestation if attestation.startswith('attest:') else f'attest:{attestation}'
        target = self.get(name.split(':', 1)[1])
        if not target or target[0]['type'] == records.IDENTITY:
            raise ValueError(f'no such attestation (identities cannot be deleted): {name}')
        if target[0]['signed_by'] != identity.identity_id:
            raise ValueError(f'{name} was not signed by {identity.username}, so you cannot delete it' if target[0]['signed_by']
                             else f'{name} is anonymous (not signed), so there is no key that can delete it')
        return self.submit(records.REVOCATION, name, {'reason': reason} if reason else {}, identity)

    def submit(self, kind, node, record, identity=None):
        """Submit a record about ``node``. ``identity`` (a ``crypto.Identity``) signs it; without
        one it is anonymous. Returns the new attestation id."""
        record = build_record(kind, node, record)
        body = {'record': record}
        if identity is not None:
            if identity.server and identity.server != self.url:
                raise ValueError(f'this identity is registered on {identity.server}, not {self.url}')
            body.update(signature=identity.sign(records.signing_payload(record)),
                        signed_by=identity.identity_id)
        return self._call('POST', 'revoke' if kind == records.REVOCATION else kind, json=body)['id']


def build_record(kind, node, content):
    """The full record that gets signed and stored: ``content`` plus its ``kind``, the ``subject`` node,
    ``attested_at`` and ``environment`` (``content`` may override the last two)."""
    record = {'kind': kind, 'subject': node, 'attested_at': records.now(),
              'environment': environment(), **content}
    return records.validate(kind, record)


# -- builders for each kind of record ---------------------------------------------

def hash_record(node, path=None, algos=None):
    """A hash record for a ``irds`` v2 node (or its name): a table is hashed by content
    (``hashing.table_hash``), a file by its bytes -- at ``path`` if given (a copy you obtained
    yourself, so ``source`` is null), else the node's own downloaded copy. A file is hashed with
    ``algos`` -- by default SHA-256 plus whatever algorithms the node declares a hash in, so the
    result can be compared with the declared one."""
    from .. import load
    from .. import vocabulary
    node = load(node) if isinstance(node, str) else node
    types = vocabulary.types_of(node)
    if any(vocabulary.is_subclass(t, 'irds:Table') for t in types):
        if path:
            raise ValueError('path only applies to a file')
        return {'hash_version': records.TABLE_HASH_VERSION, 'hashes': [hashing.table_hash(node)]}
    if any(vocabulary.is_subclass(t, 'irds:File') for t in types):
        source = None
        if not path:
            urls = [getattr(s, 'url', s) for s in node.sources]
            source = next((u for u in urls if isinstance(u, str)), None)
            path = node.path()
        algos = algos or sorted({'sha256', *getattr(node, 'hashes', {})})
        return {'source': source, 'hashes': hashing.file_hashes(path, algos)}
    raise ValueError(f'{node.qualified_name} is a {types[0]}: only files and tables can be hash-attested')


def attest_hash(node, path=None, identity=None, client=None, algos=None):
    name = node if isinstance(node, str) else node.qualified_name
    return (client or Client()).submit(records.HASH, name, hash_record(node, path, algos), identity)


def attest_visual(node, verdict, notes='', sample=(), identity=None, client=None):
    """Record that a person looked at a sample of the node's records (``sample``: the ids)."""
    record = {'verdict': verdict, 'notes': notes, 'sample': list(sample)}
    return (client or Client()).submit(records.VISUAL, node, record, identity or crypto.Identity.load())


def attest_evaluation(node, system, measures, identity=None, client=None):
    record = {'system': system, 'measures': dict(measures)}
    return (client or Client()).submit(records.EVALUATION, node, record, identity)


def revoke(attestation, reason=None, identity=None, client=None):
    return (client or Client()).revoke(attestation, reason, identity)


def for_node(node, client=None):
    return (client or Client()).for_node(node if isinstance(node, str) else node.qualified_name)
