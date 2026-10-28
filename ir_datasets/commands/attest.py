def main(args):
    # Imported here: ir_datasets.v2 is heavy and `import ir_datasets` must not pay for it.
    from ..v2.attest import cli
    return cli.main(args)
