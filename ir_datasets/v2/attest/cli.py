"""``ir_datasets attest``: submit and read attestations.

    ir_datasets attest register alice              # make an identity (signs your attestations from then on)
    ir_datasets attest hash irds:antique-docs      # hash a table or file and submit it
    ir_datasets attest visual irds:antique-docs --verdict ok --sample 2020338_0 --notes "texts look right"
    ir_datasets attest evaluation irds:antique-test --system "Terrier BM25" -m nDCG@10=0.32
    ir_datasets attest show irds:antique-test      # what has been attested (or give ids)
"""
import argparse
import json
import sys

from . import client as clientmod, crypto, records


def _identity(args):
    if getattr(args, 'anonymous', False):
        return None
    return crypto.Identity.load()


def _measure(text):
    name, sep, value = text.rpartition('=')
    try:
        return name, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{text!r}: expected NAME=NUMBER, like nDCG@10=0.32')


def _submit(args, kind, node, record):
    identity = _identity(args)
    client = clientmod.Client(args.server)
    if args.dry_run:
        print(json.dumps({'record': clientmod.build_record(kind, node, record),
                          'signed_by': identity.username if identity else None}, indent=2))
        return 0
    id = client.submit(kind, node, record, identity)
    print(f'{id}  ({kind} attestation about {node}, {"signed by " + identity.username if identity else "anonymous"})')
    return 0


def cmd_register(args):
    existing = crypto.Identity.load()
    if existing and not args.force:
        sys.exit(f'already registered as {existing.username} ({crypto.Identity.path()}); use --force to replace it')
    identity = clientmod.Client(args.server).register(args.username)
    print(f'registered {identity.username} as {identity.identity_id}; key saved to {crypto.Identity.path()}')


def cmd_hash(args):
    return _submit(args, records.HASH, args.node,
                   clientmod.hash_record(args.node, path=args.file, algos=tuple(args.algo) if args.algo else None))


def cmd_visual(args):
    if not _identity(args):
        sys.exit('visual attestations must be signed: run "ir_datasets attest register <username>" first')
    return _submit(args, records.VISUAL, args.node,
                   {'verdict': args.verdict, 'notes': args.notes, 'sample': args.sample})


def cmd_evaluation(args):
    return _submit(args, records.EVALUATION, args.node, {'system': args.system, 'measures': dict(args.measure)})


def cmd_show(args):
    client = clientmod.Client(args.server)
    out = []
    for target in args.target or [None]:
        if target is None:
            out += client.ids()
        elif ':' in target:
            out += client.for_node(target)
        else:
            out += client.get(*target.split(','))
    print(json.dumps(out, indent=2))


def parser():
    p = argparse.ArgumentParser(prog='ir_datasets attest', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--server', help=f'attestation server (default: $IR_DATASETS_ATTEST_URL or {clientmod.DEFAULT_URL})')
    sub = p.add_subparsers(dest='command', required=True)

    def submitting(name, fn, help):
        s = sub.add_parser(name, help=help)
        s.add_argument('node', help='qualified node name, e.g. irds:antique-test')
        s.add_argument('--anonymous', action='store_true', help='do not sign, even if you have an identity')
        s.add_argument('--dry-run', action='store_true', help='print the record instead of submitting it')
        s.set_defaults(fn=fn)
        return s

    s = sub.add_parser('register', help='create an identity and save its key')
    s.add_argument('username')
    s.add_argument('--force', action='store_true', help='replace a saved identity')
    s.set_defaults(fn=cmd_register)

    s = submitting('hash', cmd_hash, 'hash a file or table')
    s.add_argument('--file', help='hash this local copy of a file node (recorded as having no source)')
    s.add_argument('--algo', action='append', help='hash algorithm for files (repeatable; default: sha256 and the algorithms the node declares)')

    s = submitting('visual', cmd_visual, 'report a manual inspection (signed)')
    s.add_argument('--verdict', required=True, choices=['ok', 'issues'])
    s.add_argument('--notes', default='')
    s.add_argument('--sample', nargs='*', default=[], help='ids of the records you looked at')

    s = submitting('evaluation', cmd_evaluation, 'report scores for a system on a benchmark')
    s.add_argument('--system', required=True)
    s.add_argument('-m', '--measure', action='append', type=_measure, required=True, metavar='NAME=VALUE')

    s = sub.add_parser('show', help='print attestations: for nodes (irds:...), by id, or all ids')
    s.add_argument('target', nargs='*')
    s.set_defaults(fn=cmd_show)

    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except clientmod.AttestError as e:
        sys.exit(f'server refused: {e}')
    except ValueError as e:
        sys.exit(str(e))
