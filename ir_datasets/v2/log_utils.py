"""Generic download-event logging for v2 ``Resource``\\ s.

Every time a Resource's bytes are actually *fetched* (not merely read from
an already-local copy -- a cache hit, or a user-supplied external file),
``nodes.Resource.path``/``.stream`` call :func:`log_download` once, here,
regardless of provider: this module knows nothing about any particular
dataset family (trec-browser or otherwise) -- it only ever sees a generic
``Node`` and a filesystem path.

The log itself is one append-only, gzip-compressed JSON-lines file under the
ir_datasets home directory (``util.home_path()``), not one per provider: a
single file is simpler to tail/grep/parse across every download regardless
of where it came from, and gzip transparently supports being opened for
read after many separate appends (each ``gzip.open(..., 'at')`` call writes
its own member; the stdlib's reader already concatenates members on read).
"""
import datetime
import gzip
import hashlib
import json

import ir_datasets
from ir_datasets.util import home_path

_logger = ir_datasets.log.easy()

#: One shared log for every v2 Resource download, regardless of provider.
LOG_FILENAME = 'v2_download_log.jsonl.gz'


def log_path():
    return home_path() / LOG_FILENAME


def _file_md5(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def log_download(node, path):
    """Append one entry recording that ``node`` (a ``nodes.Resource``) was
    just downloaded to ``path`` -- called once per actual fetch, never for
    a cache hit or a user-supplied external file (see the call sites in
    ``nodes.Resource``). Records a timestamp, the node's own name, the
    *actual* md5 of what landed on disk (independent of whatever hash, if
    any, the node itself declared/verified -- see e.g. trec-browser's
    ``_SoftHashResource``, whose declared hash is only a hint), and where on
    the filesystem it is.

    Best effort: a failure here (e.g. a read-only/full home directory) is
    only logged, never raised -- a logging problem must never turn an
    already-successful download into a failure.
    """
    try:
        entry = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'node': getattr(node, 'qualified_name', None) or node.name,
            'md5': _file_md5(path),
            'path': str(path),
        }
        dest = log_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, 'at', encoding='utf-8') as fh:
            fh.write(json.dumps(entry) + '\n')
    except OSError as e:
        _logger.warn(f'could not record a download log entry for {node}: {e}')
