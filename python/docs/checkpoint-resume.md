# Checkpoint and resume a local run

Use this path only when a compatible local lifetime reduction may be interrupted.
It persists sufficient statistics in a local SQLite file; it does not store raw
input rows and it is not a distributed checkpoint service.

Keep the source revision unchanged, reopen the same local store, and provide
the remaining compatible input. A changed source revision, invalid checkpoint,
or incompatible reducer returns a typed failure rather than advancing the
checkpoint.

Run the complete example from the repository root:

```console
python python/examples/checkpoint_resume.py
```

It creates a store, commits one lifetime chunk, reopens that store, and applies
a second compatible chunk. Its verified output is:

```text
rows=2; events=1; total_time=3.75
```

Use a different architecture when multiple workers, a network filesystem, or a
distributed store is required. Review [known limits](../KNOWN_LIMITS.md) before
putting a resumed result on a production decision path.
