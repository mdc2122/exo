# MiMo MTP cluster benchmark correction - 2026-06-06

The single-process benchmark load-only exit 137 is not evidence that exo cannot
run MiMo V2.5 Pro. It only shows that one local benchmark process cannot
materialize the full quantized model by itself. The live benchmark path for MiMo
must use the exo cluster tensor-parallel execution path.

## Correct live path

1. Start or point to a running exo cluster API.
2. Collect AR baseline rows through /bench/chat/completions.
3. Add a guarded cluster MTP request path that is disabled by default.
4. Collect same-cluster MTP rows through the same API path.
5. Keep Slice 5 blocked until MTP rows beat AR rows on the same cluster.

## Harness

The cluster harness is scripts/bench_mimo_mtp_cluster.py. It does not load model
weights. It posts to an already-running exo API and emits JSON-lines rows with
cluster generation stats and power usage.

Example AR baseline command:

    uv run python scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 3 --mode-label ar --list-models

Future guarded MTP example after the worker/API flag exists:

    uv run python scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 3 --mode-label mtp-d1 --payload-extra-json '{"mimo_mtp_fastpath":true,"mimo_mtp_depth":1}'

## Current gate

No same-cluster AR-vs-MTP comparison rows exist yet. The cluster AR harness is
ready, but a guarded cluster MTP request path still needs to be wired before MTP
rows can be measured honestly through exo.
