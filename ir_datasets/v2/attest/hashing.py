"""Hashes that go into HashAttestation records."""
import dataclasses
import hashlib
import json

from .records import TABLE_HASH_VERSION


def file_hashes(path, algos=('sha256',)):
    """``["sha256:...", ...]`` of a local file's bytes."""
    hashers = [hashlib.new(a) for a in algos]
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            for h in hashers:
                h.update(chunk)
    return [f'{a}:{h.hexdigest()}' for a, h in zip(algos, hashers)]


def _jsonable(value):
    """A record value as plain JSON types. Subclasses count as their base (a ``namedtuple`` of any
    name, an ``enum`` of ints...). Anything unrecognized is an error: hashing its ``repr`` would
    make the hash depend on the Python or library version."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            return repr(value)       # NaN / inf are not JSON
        return value
    if hasattr(value, '_asdict'):
        return {k: _jsonable(v) for k, v in value._asdict().items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if hasattr(value, 'item') and callable(value.item):      # numpy scalars
        return _jsonable(value.item())
    raise TypeError(f'cannot hash a record containing {type(value).__name__}')


def table_hash(records, version=TABLE_HASH_VERSION):
    """``"sha256:..."`` over a table's records, in iteration order (version 1).

    Each record becomes one line of canonical JSON -- its fields by name (sorted keys, no
    whitespace, UTF-8, bytes as hex, sets sorted, NaN/inf as text; an unrecognized type is an error) -- and the hash is SHA-256 of the lines joined
    by ``\\n``. It depends on parsed content and order, not on the bytes the table came from, so
    re-encodings of the same data agree. Changing any of this means a new ``hash_version``."""
    if version != 1:
        raise ValueError(f'unknown table hash version {version}')
    h = hashlib.sha256()
    for record in records:
        h.update(json.dumps(_jsonable(record), sort_keys=True, separators=(',', ':'),
                            ensure_ascii=False).encode('utf-8'))
        h.update(b'\n')
    return f'sha256:{h.hexdigest()}'
