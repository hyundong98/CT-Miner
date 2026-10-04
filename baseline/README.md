# Baselines and shared utilities

| Directory | Contents |
| --- | --- |
| `ct_hash/` | Our CT-Hash baseline: independent-window Cartesian-tree encoding and hash aggregation |
| `common/` | Shared Java collection, transport, and CT validation utilities |

Run CT versus CT-Hash with `python run.py hash`; see the [main README](../README.md) for commands. The indexed CT implementation lives in `ctminer/java/`.

Third-party baseline implementations are not bundled. `sources.lock.json` and [NOTICE.md](NOTICE.md) retain historical upstream references and notices; they are not build dependencies.
