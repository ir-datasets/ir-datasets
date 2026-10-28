"""What an attestation record looks like: kinds, validation, canonical JSON and what a signature covers.

Shared by the server (validates on the way in), the client (builds records) and the provider.
"""
import datetime
import hashlib
import json
import re
import uuid

HASH, VISUAL, EVALUATION, IDENTITY, REVOCATION = 'hash', 'visual', 'evaluation', 'identity', 'revocation'
#: Kinds a client may submit to ``POST /{kind}``; identities have their own ``POST /register``.
SUBMITTABLE = (HASH, VISUAL, EVALUATION)
#: A revocation (``POST /revoke``) is not stored: it is a signed request, valid only from the identity that signed the
#: attestation it names, and the server deletes that attestation when it verifies.
KINDS = (*SUBMITTABLE, IDENTITY, REVOCATION)

#: Visual attestations are only meaningful if someone is accountable for having looked; a revocation is the
#: proof that the key that made an attestation is the one deleting it.
SIGNATURE_REQUIRED = (VISUAL, REVOCATION)

MAX_BODY_BYTES = 64 * 1024
#: ``table_hash`` version this package implements (see ``hashing.table_hash``).
TABLE_HASH_VERSION = 1

_SUBJECT = re.compile(r'^[a-z][a-z0-9_-]*:\S+$')
_ATTESTATION_NAME = re.compile(r'^attest:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
_USERNAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{1,38}$')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def new_id():
    return str(uuid.uuid4())


def signing_payload(record):
    """The bytes a signature covers: the whole record as canonical JSON. The record carries its
    own ``kind`` and ``subject``, so the signature is bound to them too."""
    return canonical_json(record).encode('utf-8')


def _timestamp(value):
    try:
        datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except (TypeError, ValueError):
        raise ValueError('attested_at must be a UTC ISO 8601 timestamp like 2026-10-28T12:00:00Z')


def _hashes(values):
    if not isinstance(values, list) or not values:
        raise ValueError('hashes must be a non-empty list of "algo:hexdigest" strings')
    for spec in values:
        algo, sep, digest = spec.partition(':') if isinstance(spec, str) else ('', '', '')
        try:
            size = hashlib.new(algo).digest_size if sep else None
        except ValueError:
            size = None
        if size is None:
            raise ValueError(f'bad hash {spec!r}: expected "algo:hexdigest" with a known algorithm')
        if not re.fullmatch(r'[0-9a-f]+', digest) or (size and len(digest) != size * 2 and algo not in ('shake_128', 'shake_256')):
            raise ValueError(f'bad hash {spec!r}: digest is not lowercase hex of the right length')


def validate(kind, record):
    """Check ``record`` is well-formed for ``kind``; returns it. Raises ``ValueError``.

    Only structure is checked. Whether a hash is *right* is for readers to judge by comparing
    attestations, not for the server."""
    if kind not in KINDS:
        raise ValueError(f'unknown attestation kind {kind!r}')
    if not isinstance(record, dict):
        raise ValueError('record must be a JSON object')
    if record.get('kind') != kind:
        raise ValueError(f'record kind must be {kind!r}')
    if kind == IDENTITY:
        if record.get('subject') is not None:
            raise ValueError('an identity has no subject')
    else:
        validate_subject(record.get('subject'))
    _timestamp(record.get('attested_at'))
    env = record.get('environment', {})
    if not isinstance(env, dict):
        raise ValueError('environment must be an object')
    if kind == HASH:
        _hashes(record.get('hashes'))
        if 'hash_version' in record:     # a table hash
            if not isinstance(record['hash_version'], int) or isinstance(record['hash_version'], bool):
                raise ValueError('hash_version must be an integer')
        elif record.get('source') is not None and not isinstance(record['source'], str):
            raise ValueError('source must be a URL string, or null for a local file')
    elif kind == VISUAL:
        if record.get('verdict') not in ('ok', 'issues'):
            raise ValueError('verdict must be "ok" or "issues"')
        if not isinstance(record.get('notes', ''), str):
            raise ValueError('notes must be a string')
        sample = record.get('sample', [])
        if not isinstance(sample, list) or not all(isinstance(s, (str, int)) for s in sample):
            raise ValueError('sample must be a list of the ids that were inspected')
    elif kind == EVALUATION:
        if not isinstance(record.get('system'), str) or not record['system'].strip():
            raise ValueError('system must be a non-empty string')
        measures = record.get('measures')
        if (not isinstance(measures, dict) or not measures or not all(
                isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
                for k, v in measures.items())):
            raise ValueError('measures must be a non-empty object of measure name -> number')
    elif kind == REVOCATION:
        if not _ATTESTATION_NAME.match(record['subject']):
            raise ValueError('a revocation\'s subject must be the attestation it deletes, like "attest:<id>"')
        reason = record.get('reason', '')
        if not isinstance(reason, str) or len(reason) > 500:
            raise ValueError('reason must be a string of at most 500 characters')
    elif kind == IDENTITY:
        if not _USERNAME.match(record.get('username') or ''):
            raise ValueError('username must be 2-39 characters: letters, digits, "_", "." or "-"')
        if not isinstance(record.get('public_key'), str) or not record['public_key']:
            raise ValueError('public_key is required')
    return record


def validate_subject(subject):
    if not isinstance(subject, str) or not _SUBJECT.match(subject) or len(subject) > 300:
        raise ValueError('subject must be a qualified node name like "irds:antique-test"')
    return subject
