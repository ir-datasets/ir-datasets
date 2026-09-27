"""Where a Resource's bytes come from, and the stream pipeline over them.

A ``Resource`` node names a set of bytes and a list of ``Source``s -- places
that can supply them (a URL, the community mirror, a manual/local copy).
Getting from those bytes to something a parser can read is a *pipeline* --
extract a member from an archive, decompress it, apply a fixup -- expressed
declaratively::

    archive.member('collection.tsv').gunzip().pipe(fix_encoding)

Pipeline steps are expressions, not nodes: they have no name of their own. If a
step's output is worth addressing or sharing, make it a named node instead.

Everything a pipeline step produces implements the same tiny interface the v1
parsers already expect (``path(force=True)`` / ``stream()``, see ``Readable``
below), so all of ``ir_datasets.formats`` works unchanged.
"""
import contextlib
import io
import os
import shutil
from pathlib import Path

import ir_datasets
from ir_datasets.util import (
    Bz2Extract, Cache, GzipExtract, IterStream, Lz4Extract, RelativePath,
    TarExtract, ZipExtract,
)
from ir_datasets.util.download import (
    Download, GoogleDriveDownload, LocalDownload, RequestsDownload,
)

_logger = ir_datasets.log.easy()

IRDS_MIRROR = 'https://mirror.ir-datasets.com/'

#: Bumped when a docstore's on-disk format changes. It is part of the cache
#: path, so a new format writes a new file instead of colliding with the old
#: one, and "is my cache stale?" never requires guessing.
DOCSTORE_FORMAT = 1

_LEGACY_CACHE = None


def as_source(location):
    """Accept a bare URL string wherever a Source is expected."""
    if isinstance(location, str):
        return Source(location)
    return location


def materialize(source, md5):
    """Resolve a ``Source.irds()`` placeholder into a real ``Source(url)``.

    ``Source.irds()`` is written at a Resource's construction site, before
    that Resource's own md5 (also a constructor arg) is known -- so it can't
    build its mirror URL itself; it's a placeholder, not yet a real source.
    A ``Resource`` calls this once its own ``self.md5`` is settled, so what
    ends up in ``self.sources`` is an ordinary, concrete ``Source`` like any
    other -- its ``.url``, ``repr()``, and downstream consumers (the webapp,
    ``local_copy_hint``) see the actual mirror URL, not an opaque marker.
    Anything that isn't an unresolved placeholder passes through unchanged.
    """
    if isinstance(source, _IrdsSource) and md5:
        return Source(f'{IRDS_MIRROR}{md5}')
    return source


def default_cache_path(provider, name):
    """Where a Resource's bytes are cached: ``<home>/<provider>/<name>``.
    Never written by hand.

    Human-readable and naturally partitioned by provider -- a single flat
    directory of hashes doesn't scale as a *browsable* thing, even if
    filesystems handle the entry count fine. Hierarchical for free wherever
    ``name`` itself contains ``/`` (as e.g. ``hf:`` names do), since a path
    join treats it as a separator. ``provider`` is ``None`` for an ad hoc
    Resource that was never registered through a ``Provider`` (a local test
    fixture, a one-off script); those land in a fixed ``_local`` bucket
    instead of a real provider's own directory.

    Not content-addressed by md5 (v1's convention): two differently-named
    Resources that happen to share identical bytes are no longer deduplicated
    on disk -- traded away for readability, and because the md5-keyed
    ``downloads/`` layout still matters for exactly one thing, reading an
    *existing* v1 cache (see ``legacy_path``/``migrate_legacy``), which this
    function has nothing to do with. Integrity is still checked at ``verify``
    time regardless of where the file lives.
    """
    home = Path(ir_datasets.util.home_path())
    return home / (provider or '_local') / name.lstrip('_')


def local_copy_hint(file):
    """Message telling the user where to symlink a copy they already have.

    v1 surfaced this via a ``LocalDownload`` source (and ``skip_local`` opted
    out). v2 does not need the extra source -- the content-addressed cache path
    *is* that location -- but the hint is still worth printing before a large
    download starts. Returns None when it would not be useful.
    """
    if not file.md5:
        return None
    small_file_size = int(os.environ.get('IR_DATASETS_SMALL_FILE_SIZE', '5000000'))
    if file.size is not None and file.size < small_file_size:
        return None
    url = next((m.url for m in file.sources if getattr(m, 'url', None)), None)
    subject = f'a copy of {url}' if url else 'this file'
    return (f'If you already have {subject}, you can symlink it here to avoid '
            f'downloading it again: {file.cache_path}')


def _legacy_cache():
    """md5 -> v1 cache path, generated from v1's downloads.json by migrate.py."""
    global _LEGACY_CACHE
    if _LEGACY_CACHE is None:
        path = Path(__file__).parent / 'legacy_cache.json'
        if path.exists():
            import json
            with open(path) as fin:
                _LEGACY_CACHE = json.load(fin)
        else:
            _LEGACY_CACHE = {}
    return _LEGACY_CACHE


def legacy_path(md5):
    """An existing v1 cache file for these bytes, if the user already has one.

    Never re-downloaded -- this is what keeps a v2 upgrade from re-fetching
    terabytes. What happens to the file once found is the caller's choice:
    ``Resource.existing_path`` moves it to its v2 location via
    ``migrate_legacy`` (see there), the one time it's found; a docstore's own
    legacy check reads it in place instead, since a docstore's *derived*
    artifact sits right next to the raw file either way.
    """
    if not md5:
        return None
    rel = _legacy_cache().get(md5)
    if rel is None:
        return None
    candidate = Path(ir_datasets.util.home_path()) / rel
    return candidate if candidate.exists() else None


def migrate_legacy(legacy, cache_path):
    """Move a v1 cache file (or directory -- a v1 docstore is one) to its v2
    location, once, the first time it's found (see ``Resource.existing_path``/
    ``DocTable.docstore_path``) -- not a separate migration step the user has to
    remember to run, since there's no reason to make them: nothing else needs
    the old v1 layout left untouched, and every future lookup (this run's, or
    a later one's) should hit ``cache_path`` directly without consulting
    ``legacy_path`` again.

    A symlink is left at the old v1 path pointing at the new one, best-effort
    -- so anything still hard-coded to the v1 location (an old script, a
    notebook) keeps working. The symlink is a courtesy, not load-bearing: if
    it can't be created (no permission, an unsupported filesystem), the move
    itself -- the part that actually matters -- still happened.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    was_dir = legacy.is_dir()
    shutil.move(str(legacy), str(cache_path))
    _logger.info(f'migrated v1 cache {legacy} -> {cache_path}')
    try:
        legacy.symlink_to(cache_path, target_is_directory=was_dir)
    except OSError:
        pass
    return cache_path


class Source:
    """One place a Resource's bytes can be fetched from.

    ``Source.irds()`` is the ir-datasets community mirror, addressed by the
    Resource's md5 -- so it can only be used on a Resource that declares one.
    ``Source.local()`` is not a URL at all: it's a file the user must obtain
    themselves (a signed agreement, a paid corpus, ...) -- see
    ``PLAN_V2_SITE.md``'s "Files that can't be automatically obtained".
    """

    def __init__(self, url, headers=None, auth=None, cookies=None):
        self.url = url
        self.headers = headers
        self.auth = auth
        self.cookies = cookies

    @classmethod
    def irds(cls):
        return _IrdsSource()

    @classmethod
    def local(cls, path, instructions=None):
        """A file the user must obtain themselves (signed agreement, etc)."""
        return _ManualSource(path, instructions)

    def _build(self, file):
        kwargs = {}
        if self.headers:
            kwargs['headers'] = self.headers
        if self.auth:
            kwargs['auth'] = self.auth
        if self.cookies:
            kwargs['cookies'] = self.cookies
        if self.url.startswith('https://drive.google.com/'):
            return GoogleDriveDownload(self.url, **kwargs)
        return RequestsDownload(self.url, **kwargs)

    def __repr__(self):
        return f'Source({self.url!r})'


class _IrdsSource(Source):
    """An unresolved placeholder: ``Source.irds()`` is written before a
    Resource's md5 is known (it's a constructor arg the source list is built
    from), so it cannot address the mirror URL itself. ``materialize()``
    replaces it with a real ``Source(url)`` as soon as a Resource has an md5
    to key off -- see that function's docstring. This class only remains
    reachable when a Resource declares ``Source.irds()`` with no md5 at all
    (a genuine authoring mistake), in which case ``_build`` still raises,
    now purely as a fallback rather than the normal path.
    """
    def __init__(self):
        super().__init__(url=None)

    def _build(self, file):
        raise ValueError(
            f'{file.name}: Source.irds() requires an md5 (the mirror is '
            f'addressed by content hash)')

    def __repr__(self):
        return 'Source.irds()'


class _ManualSource(Source):
    def __init__(self, path, instructions):
        super().__init__(url=None)
        self.local_path = path
        self.instructions = instructions

    def _build(self, file):
        msg = self.instructions
        if msg:
            msg = msg.format(path=self.local_path)
        return LocalDownload(self.local_path, msg, mkdir=False)

    def __repr__(self):
        return f'Source.local({self.local_path!r})'


class Readable:
    """Anything a parser can read: a Resource, or a pipeline over one."""

    def path(self, force=True):
        raise NotImplementedError

    @contextlib.contextmanager
    def stream(self):
        raise NotImplementedError

    # -- pipeline ops (each returns a new, unnamed Readable) ----------------

    def member(self, path, compression='gz'):
        """Extract one member from a tar archive."""
        return _Pipe(self, TarExtract(self, path, compression=compression),
                     f'member({path!r})')

    def zip_member(self, path):
        return _Pipe(self, ZipExtract(self, path), f'zip_member({path!r})')

    def relative(self, path):
        return _Pipe(self, RelativePath(self, path), f'relative({path!r})')

    def gunzip(self):
        return _Pipe(self, GzipExtract(self), 'gunzip()')

    def bunzip2(self):
        return _Pipe(self, Bz2Extract(self), 'bunzip2()')

    def unlz4(self):
        return _Pipe(self, Lz4Extract(self), 'unlz4()')

    def pipe(self, transform):
        """Apply a user-defined stream transform (see @transform)."""
        return _Pipe(self, transform(self), f'pipe({transform.name})')

    def cache(self, path):
        """Materialize this pipeline's output to `path` once, then reuse it.

        Needed when a parser wants a real file (random access / docstore
        building) rather than a one-shot stream.
        """
        return _Pipe(self, Cache(self, Path(path)), f'cache({path!r})', cached=True)


class _Pipe(Readable):
    def __init__(self, parent, impl, label, cached=False):
        self._parent = parent
        self._impl = impl
        self._label = label
        self._cached = cached

    def path(self, force=True):
        if hasattr(self._impl, 'path'):
            return self._impl.path(force)
        raise TypeError(
            f'{self!r} is a stream-only pipeline; add .cache(path) if a '
            f'concrete file is required')

    @contextlib.contextmanager
    def stream(self):
        with self._impl.stream() as stream:
            yield stream

    def __repr__(self):
        return f'{self._parent!r}.{self._label}'


class transform:
    """Decorator making a generator function usable as a pipeline step.

    The wrapped function takes a binary stream and yields bytes::

        @transform
        def fix_encoding(stream):
            for line in stream:
                yield line.replace(b'\\xc3\\x83', b'\\xc3')
    """

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, source):
        return _TransformStream(source, self.fn)

    def __repr__(self):
        return f'<transform {self.name}>'


class _TransformStream:
    def __init__(self, source, fn):
        self._source = source
        self._fn = fn

    @contextlib.contextmanager
    def stream(self):
        with self._source.stream() as stream:
            yield io.BufferedReader(
                IterStream(iter(self._fn(stream))),
                buffer_size=io.DEFAULT_BUFFER_SIZE)


def build_download(file):
    """Turn a Resource node's declared sources into a v1 ``Download``.

    Mirrors v1 behaviour, including offering a local symlink path for large
    files with a known md5 so users can avoid re-downloading.
    """
    # v1 added a LocalDownload source pointing at downloads/<md5> so a user could
    # symlink a copy they already had (and `skip_local` opted out of it). v2 needs
    # neither: the content-addressed cache path *is* that location, so such a
    # symlink is simply a cache hit.
    downloads = [source._build(file) for source in file.sources]
    return Download(
        downloads,
        expected_md5=file.md5,
        cache_path=str(file.cache_path) if file.cache_path else None,
        dua=file.dua,
        size_hint=file.size,
    )
