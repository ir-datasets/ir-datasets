"""The ``documentation:`` provider: community-edited Markdown pages about nodes.

It defines **no nodes**. It only contributes properties and edges *about* nodes
other providers own, which is what keeps it safe to let anyone edit: a page can be
wrong, but it cannot change what a node downloads, checks, or evaluates.

Pages live in their own repository, https://github.com/ir-datasets/documentation (Markdown,
one file per described node, at ``<prefix>/<name>.md``)::

    irds/mast.md           documents irds:mast
    hf/owner/repo.md       documents hf:owner/repo

Updates are **live**: the first time a ``documentation:`` property is read in a process, the
branch's tarball is fetched (a conditional request; cached under ``<ir_datasets home>/documentation``
and not re-checked for an hour). The network is never required: if it is down, the cached copy is
used (with one warning), and with no cache there is simply no documentation. Configuration:

* ``IR_DATASETS_DOCUMENTATION=<dir>`` reads a local checkout instead (for editing and previewing);
* ``IR_DATASETS_DOCUMENTATION_REF=<branch|tag|sha>`` freezes the ref (default ``main``); a full
  commit sha is fetched once and never refreshed;
* ``IR_DATASETS_DOCUMENTATION_REPO=<owner/name>`` points at a fork;
* ``IR_DATASETS_DOCUMENTATION_TTL=<seconds>`` is how long a fetched copy is trusted (default an hour;
  ``0`` asks GitHub every time, and a ``304`` costs one small request). ``refresh()`` does that on demand.

A file is YAML frontmatter plus a Markdown body::

    ---
    description: One sentence that says what this is, for listings.
    ---
    The page: any amount of Markdown ...

A node that needs only a description (a family of near-identical benchmarks, say) gets an entry in
``<prefix>/_index.yaml`` instead of a page: a mapping from the node's name to its description, or to a
mapping of the same keys as the frontmatter. An entry yields exactly the frontmatter properties and no
``documentation:about``. Where a node has both, the page wins outright: its entry in the index is ignored (``lint`` says it can go), not merged. Each node reports the one file it was read from, as ``documentation:source``.

The frontmatter keys become ``documentation:<key>`` literals (the declared set below; an
unknown key is a lint error) and the body becomes ``documentation:about``. Links in the
body are standard Markdown, ``[text](target)``; a target with a provider's prefix
(``irds:…``, ``hf:…``, ``documentation:…``) is a link to that node and is recorded as a
``documentation:links`` edge, which is also how backlinks work. ``dblp:``, ``doi:``,
``http:`` and ``https:`` are accepted and not edges. Relative targets and ``#section``
anchors are not supported -- every target is fully qualified.
"""
import io
import json
import os
import re
import tarfile
import time
import warnings
from pathlib import Path

from .base import Literal
from .registry import ManifestProvider

ENV = 'IR_DATASETS_DOCUMENTATION'
ENV_REF = 'IR_DATASETS_DOCUMENTATION_REF'
ENV_REPO = 'IR_DATASETS_DOCUMENTATION_REPO'
ENV_TTL = 'IR_DATASETS_DOCUMENTATION_TTL'
REPO = 'ir-datasets/documentation'
REF = 'main'
#: The branch edits are proposed against (the default `REF`; a frozen ref is read-only).
BRANCH = 'main'
#: How long a fetched copy is used without asking GitHub whether it changed.
TTL_SECONDS = 3600
TIMEOUT_SECONDS = 10

#: Frontmatter keys a page may set, i.e. the literal ``documentation:*`` properties.
FRONTMATTER_KEYS = {
    'description': 'a short description of the node, for listings: one sentence',
}

#: Schemes a link may use without being an edge to a node.
EXTERNAL_SCHEMES = {'http', 'https', 'dblp', 'doi'}

_FRONTMATTER = re.compile(r'\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)', re.S)
# [text](target), skipping images and ignoring an optional "title" after the target.
_LINK = re.compile(r'(?<!!)\[(?:[^\]\\]|\\.)*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
_FENCE = re.compile(r'^(```|~~~).*?^\1', re.S | re.M)


_SHA = re.compile(r'^[0-9a-f]{40}$')
_fetched = {}       # (repo, ref) -> content dir or None, per process


def _cache_root(repo, ref):
    from ir_datasets.util import home_path
    return home_path() / 'documentation' / repo.replace('/', '--') / re.sub(r'[^A-Za-z0-9._-]', '_', ref)


def _extract(data, dest):
    """Write the pages (``*.md``) and ``_index.yaml`` files of a GitHub tarball (``<repo>-<ref>/...``)
    under ``dest``. Only regular files of those kinds are read, and only by a path that stays inside
    ``dest``."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for member in tar:
            parts = Path(member.name).parts[1:]
            if not (member.isfile() and parts and (parts[-1].endswith('.md') or parts[-1] == '_index.yaml')) \
                    or '..' in parts:
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())


def _ttl():
    try:
        return float(os.environ[ENV_TTL])
    except (KeyError, ValueError):
        return TTL_SECONDS


def fetch(repo=None, ref=None, *, ttl=None):
    """The directory holding the live pages, refreshed if it is older than ``ttl``; ``None`` if
    there are none (offline, never fetched). Never raises on a network problem."""
    import shutil
    import requests
    ttl = _ttl() if ttl is None else ttl
    repo = repo or os.environ.get(ENV_REPO) or REPO
    ref = ref or os.environ.get(ENV_REF) or REF
    root = _cache_root(repo, ref)
    content, meta_path = root / 'content', root / 'meta.json'
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            pass
    have = content.is_dir() and bool(meta)
    age = time.time() - meta.get('checked', 0)
    if have and (age < ttl or _SHA.match(ref)):
        return content
    headers = {'If-None-Match': meta['etag']} if have and meta.get('etag') else {}
    try:
        response = requests.get(f'https://codeload.github.com/{repo}/tar.gz/{ref}',
                                headers=headers, timeout=TIMEOUT_SECONDS)
        if response.status_code == 304:
            meta['checked'] = time.time()
        elif response.status_code == 200:
            tmp = root / f'content.{os.getpid()}.tmp'
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            _extract(response.content, tmp)
            shutil.rmtree(content, ignore_errors=True)
            tmp.rename(content)
            meta = {'etag': response.headers.get('ETag'), 'checked': time.time()}
        else:
            raise RuntimeError(f'HTTP {response.status_code}')
        root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta))
    except Exception as e:
        warnings.warn(f'could not refresh the documentation from {repo}@{ref} '
                      f'({"using the cached copy" if have else "none cached"}): {e}')
    return content if content.is_dir() else None


def refresh():
    """Ask GitHub now whether the pages changed (whatever the TTL), and use the result for the rest of
    the process. For a snapshot, like building the site's graph, that must not lag a recent edit."""
    _fetched.clear()
    key = (os.environ.get(ENV_REPO) or REPO, os.environ.get(ENV_REF) or REF)
    _fetched[key] = fetch(*key, ttl=0)
    return _fetched[key]


def documentation_dir():
    """The directory pages are read from: ``$IR_DATASETS_DOCUMENTATION`` if set, else the live
    copy (fetched once per process)."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV])
    key = (os.environ.get(ENV_REPO) or REPO, os.environ.get(ENV_REF) or REF)
    if key not in _fetched:
        _fetched[key] = fetch(*key)
    return _fetched[key]


def parse(text):
    """``(frontmatter dict, body)`` of a page's text."""
    from ir_datasets.lazy_libs import yaml
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta = yaml().safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError('frontmatter must be a YAML mapping')
    return meta, text[m.end():]


def link_targets(body):
    """Every link target in a page body (code fences excluded), in order."""
    return _LINK.findall(_FENCE.sub('', body))


def _scheme(target):
    head, sep, _ = target.partition(':')
    return head if sep else None


class DocumentationProvider(ManifestProvider):
    def __init__(self, prefix='documentation', *, directory=None):
        super().__init__(prefix, distribution='ir_datasets')
        self._directory = directory
        #: The root of everything this provider says about a node, so "all of a node's documentation" is
        #: one subproperty query. Never used on a node itself.
        root = self.declare_graph_property(
            'documentation', range='literal', parent='irds:metadata',
            desc='the parent of every documentation property: what the community-edited documentation '
                 'says about a node')
        self.declare_graph_property('description', range='literal', parent=root,
                                    desc=FRONTMATTER_KEYS['description'])
        self.declare_graph_property('about', range='literal', parent=root,
                                    desc='what the node is, in Markdown: the body of its documentation page')
        self.declare_graph_property('source', range='literal', parent=root,
                                    desc='the file in the documentation repository the node\'s documentation was read from: '
                                         'its page if it has one, else its entry in the provider\'s index')
        self.declare_graph_property('links', parent=root,
                                    desc='a node -> a node its documentation page links to')

    @property
    def directory(self):
        """Where pages are read from (``None`` if there are none); fetched on first use."""
        return Path(self._directory) if self._directory else documentation_dir()

    # -- reading pages --------------------------------------------------------

    def pages(self):
        """``{qualified node name: path}`` for every page in the checkout."""
        root = self.directory
        if root is None or not root.is_dir():
            return {}
        out = {}
        for path in sorted(root.glob('*/**/*.md')):
            rel = path.relative_to(root).with_suffix('')
            out[f'{rel.parts[0]}:{"/".join(rel.parts[1:])}'] = path
        return out

    def index(self):
        """``{qualified node name: {key: value}}`` from every ``<prefix>/_index.yaml``. An entry is
        a description, or a mapping of frontmatter keys; a malformed file is skipped, with a
        warning (``lint`` reports it)."""
        return {name: entry for name, (entry, _) in self._index_with_files().items()}

    def _index_with_files(self):
        """``{qualified node name: (entry, repo-relative path of the index file)}``."""
        root = self.directory
        out = {}
        for path in sorted(root.glob('*/_index.yaml')) if root is not None and root.is_dir() else ():
            try:
                for name, entry in self._read_index(path).items():
                    out[f'{path.parent.name}:{name}'] = (entry, path.relative_to(root).as_posix())
            except Exception as e:
                warnings.warn(f'documentation index {path} is unreadable and was skipped: {e}')
        return out

    @staticmethod
    def _read_index(path):
        from ir_datasets.lazy_libs import yaml
        data = yaml().safe_load(path.read_text(encoding='utf8')) or {}
        if not isinstance(data, dict):
            raise ValueError('an index must be a YAML mapping of node name to description')
        return {str(name): ({'description': entry} if isinstance(entry, str) else entry)
                for name, entry in data.items()}

    def read(self, name):
        """``(frontmatter, body)`` of the page for ``name``; KeyError if there is none."""
        path = self.pages().get(name)
        if path is None:
            raise KeyError(f'no documentation page for {name!r}')
        return parse(path.read_text(encoding='utf8'))

    def lint(self, known_prefixes=None):
        """Problems with the pages, as strings: unknown frontmatter keys, relative or
        ``#section`` links, links to an unregistered scheme, pages for nodes that do not
        exist, and an empty body. ``known_prefixes`` defaults to the default graph's."""
        from .graph import default_graph
        graph = default_graph()
        prefixes = set(graph.providers) if known_prefixes is None else set(known_prefixes)
        problems = []
        pages = self.pages()
        root = self.directory
        index = {}
        for path in sorted(root.glob('*/_index.yaml')) if root is not None and root.is_dir() else ():
            try:
                entries = self._read_index(path)
            except Exception as e:
                problems.append(f'{path.parent.name}/_index.yaml: unreadable index: {e}')
                continue
            for name, entry in entries.items():
                qualified = f'{path.parent.name}:{name}'
                index[qualified] = entry
                if not isinstance(entry, dict):
                    problems.append(f'{qualified}: index entry must be a description or a mapping')
                    continue
                for key in sorted(set(entry) - set(FRONTMATTER_KEYS)):
                    problems.append(f'{qualified}: unknown index key {key!r}')
                if not str(entry.get('description') or '').strip():
                    problems.append(f'{qualified}: index entry has no description')
                if qualified in pages:
                    problems.append(f'{qualified}: has both an index entry and a page; '
                                    f'the page wins, so the entry can go')
                if known_prefixes is None:
                    self._check_exists(graph, qualified, problems, 'index entry')
        for name, path in pages.items():
            try:
                meta, body = parse(path.read_text(encoding='utf8'))
            except Exception as e:
                problems.append(f'{name}: unreadable page: {e}')
                continue
            for key in sorted(set(meta) - set(FRONTMATTER_KEYS)):
                problems.append(f'{name}: unknown frontmatter key {key!r}')
            if 'description' not in meta:
                problems.append(f'{name}: no description in the frontmatter'
                                + ('' if body.strip() else ' (and no body: an empty page)'))
            if known_prefixes is None:
                self._check_exists(graph, name, problems, 'page')
            for target in link_targets(body):
                scheme = _scheme(target)
                if '#' in target and scheme not in ('http', 'https'):
                    problems.append(f'{name}: {target}: section anchors are not supported')
                elif scheme is None:
                    problems.append(f'{name}: {target}: relative link; use a qualified target')
                elif scheme not in EXTERNAL_SCHEMES and scheme not in prefixes:
                    problems.append(f'{name}: {target}: unknown scheme {scheme!r}')
        return problems

    @staticmethod
    def _check_exists(graph, name, problems, what):
        try:
            graph.provider_for(name)
            graph[name]
        except Exception:
            problems.append(f'{name}: {what} describes a node that does not exist')

    # -- the graph ------------------------------------------------------------

    def _literals(self, name, meta, source):
        """The frontmatter properties of ``meta`` and the one file they came from."""
        for key in FRONTMATTER_KEYS:
            if meta.get(key):
                yield name, self.qualify(key), Literal(str(meta[key]).strip())
        yield name, self.qualify('source'), Literal(source)

    def discover_edges(self):
        entries = self._index_with_files()
        pages = {}
        root = self.directory
        for name, path in self.pages().items():
            try:
                pages[name] = (*parse(path.read_text(encoding='utf8')), path.relative_to(root).as_posix())
            except Exception as e:
                # One bad page must not take the graph down (or blank the node: its index entry, if
                # any, still stands); `lint()` reports it.
                warnings.warn(f'documentation page {path} is unreadable and was skipped: {e}')
        # A page replaces the node's index entry outright; it is not merged with it.
        for name, (entry, file) in entries.items():
            if name not in pages and isinstance(entry, dict):
                yield from self._literals(name, entry, file)
        for name, (meta, body, page_file) in pages.items():
            yield from self._literals(name, meta, page_file)
            if body.strip():
                yield name, self.qualify('about'), Literal(body.strip())
            seen = set()
            for target in link_targets(body):
                scheme = _scheme(target)
                if scheme in EXTERNAL_SCHEMES or scheme is None or '#' in target or target in seen:
                    continue
                seen.add(target)
                yield name, self.qualify('links'), target
        yield from self.property_triples()


# -- where to send someone who wants to edit ------------------------------------

def repository():
    """``owner/name`` of the documentation repository (``$IR_DATASETS_DOCUMENTATION_REPO``)."""
    return os.environ.get(ENV_REPO) or REPO


def edit_url(path):
    """The GitHub page that edits ``path`` (repo-relative, as ``documentation:source`` reports it)."""
    return f'https://github.com/{repository()}/edit/{BRANCH}/{path}'


#: GitHub caps what a pre-filled URL can carry; this is comfortably under it.
MAX_TEMPLATE_CHARS = 3000


def index_path(name):
    """The repo-relative path of the index that would hold the entry for a qualified node name."""
    return f'{name.split(":", 1)[0]}/_index.yaml'


def index_line(name, description=None):
    """The line to add to ``index_path(name)`` for ``name``: its local name and a description to fill in."""
    local = name.split(':', 1)[1]
    key = local if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', local) else json.dumps(local)
    return f'{key}: {json.dumps(description or "One sentence saying what this is.")}'


def page_path(name):
    """The repo-relative path of the page for a qualified node name."""
    prefix, local = name.split(':', 1)
    return f'{prefix}/{local}.md'


def template(name, *, description=None, parts=()):
    """The text of a new page for ``name``: frontmatter, and a skeleton the author fills in."""
    description = (description or 'One sentence saying what this is, for listings.')[:300]
    front = [f'description: {json.dumps(description)}']
    body = ['What this is and why it exists, in a paragraph or two. Link other nodes as '
            '`[text](irds:node)`; every target is fully qualified.', '']
    if parts:
        body += ['## What is in it', '']
        body += [f'- [{p}]({p})' for p in list(parts)[:15]]
        if len(parts) > 15:
            body.append(f'- … and {len(parts) - 15} more')
        body.append('')
    body += ['## Things to know', '', '- ']
    text = '---\n' + '\n'.join(front) + '\n---\n' + '\n'.join(body) + '\n'
    return text if len(text) <= MAX_TEMPLATE_CHARS else text[:MAX_TEMPLATE_CHARS].rsplit('\n', 1)[0] + '\n'


def new_page_url(name, *, description=None, parts=()):
    """A GitHub link that opens a *new* page for ``name`` with the file name and a skeleton already
    filled in, ready to edit and propose."""
    from urllib.parse import quote
    directory, _, filename = page_path(name).rpartition('/')
    text = template(name, description=description, parts=parts)
    return (f'https://github.com/{repository()}/new/{BRANCH}/{quote(directory)}'
            f'?filename={quote(filename)}&value={quote(text)}')


documentation = DocumentationProvider()
