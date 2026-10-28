# attest

Attestations: records that someone hashed, inspected or evaluated a dataset node. They live on a server
(`attest.ir-datasets.com`; code in `ir-datasets.com/attest`) and appear in the graph as the `attest:` provider.

## Graph

| | |
|---|---|
| `attest:Attestation` | base type; subtypes `HashAttestation`, `VisualAttestation`, `EvaluationAttestation`, `IdentityAttestation` |
| `<node> attest:attestation attest:<id>` | a node was attested (informational edge) |
| `attest:<id> a attest:<Kind>Attestation`, `attest:record '{json}'` | the record: JSON; always has `kind`, `subject`, `attested_at`, optional `environment` |
| `attest:signature`, `attest:signedBy attest:<identity>`, `attest:verified` | signing |

Records: hash `{"source": url|null, "hashes": ["algo:hex"]}` (file) or `{"hash_version": 1, "hashes": [...]}` (table, see `hashing.table_hash`); visual `{"verdict": "ok"|"issues", "notes", "sample": [ids]}` (must be signed); evaluation `{"system", "measures": {name: number}}`; identity `{"username", "public_key"}`. A signature (Ed25519) covers the record as canonical JSON; since the record carries its own `kind` and `subject`, those are covered too.

Discovery downloads every attestation, incrementally (records never change, so after the first fetch only newer ones are requested, at most hourly; cached under `~/.ir_datasets/attest/`). Resolving a dataset node (`load`) never triggers it; listing and edge queries do. `IR_DATASETS_ATTEST_OFFLINE=1` never touches the network; `IR_DATASETS_ATTEST_URL` points at another server.

## Submit

```
ir_datasets attest register alice     # key pair; public half registered, private saved to ~/.ir_datasets/attest/identity.json
ir_datasets attest hash irds:antique-docs [--file local-copy] [--anonymous] [--dry-run]
ir_datasets attest visual irds:antique-docs --verdict ok --sample d1 d2 --notes "..."
ir_datasets attest evaluation irds:antique-test --system "Terrier BM25" -m nDCG@10=0.32
ir_datasets attest show irds:antique-test
```

```python
from ir_datasets.v2 import attest
attest.attest_hash('irds:antique-docs')
attest.attest_evaluation('irds:antique-test', 'Terrier BM25', {'nDCG@10': 0.32}, identity=attest.Identity.load())
attest.for_node('irds:antique-test')
```

Tests: `python -m unittest test.v2_attest` (no network; the server has its own).
