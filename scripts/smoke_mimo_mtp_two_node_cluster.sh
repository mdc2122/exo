#!/usr/bin/env bash
set -euo pipefail

# Live smoke gate for the Studio1+Studio2 exo MiMo MTP path.
# Non-destructive by design: it never starts, stops, restarts, pkills, rsyncs, or mutates exo.
# Live API requests require an explicit --live flag so local gates can validate the harness safely.

API_BASE="${API_BASE:-http://127.0.0.1:52415}"
MODEL_ID="${MODEL_ID:-kernelpool/MiMo-V2.5-Pro-6bit}"
MAX_TOKENS="${MAX_TOKENS:-16}"
MIN_MTP_TOK_S="${MIN_MTP_TOK_S:-25}"
OUT_DIR="${OUT_DIR:-.goose-ultrawork/evidence/live-two-node-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
PROMPT="${PROMPT:-Reply with one concise sentence proving this live exo MiMo MTP smoke reached the model.}"
SIDECAR_PATH="${MIMO_MTP_SIDECAR_PATH:-}"
# Use the repo-managed Python by default; macOS /usr/bin/python3 is too old for this codebase.
# Override with e.g. PYTHON_CMD="python3.13" if needed.
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD:-env -u VIRTUAL_ENV uv run --no-sync python})

print_checklist() {
  cat <<'EOF'
Live two-node MiMo MTP smoke checklist (non-destructive; no restarts):
1. First run the local-only gate from the repo root:

   bash scripts/smoke_mimo_mtp_two_node_cluster.sh --local-gate

2. Use the live gate only after code merges/syncs are present on both Studio1 and Studio2.
3. Reuse an already-running exo cluster; do not stop/restart exo unless Morley explicitly approved that restart.
4. Point API_BASE at the elected/live cluster API (default: http://127.0.0.1:52415 on Studio2).
5. Confirm /state has at least two nodes before trusting throughput.
6. Explicit live execution command:

   API_BASE=http://127.0.0.1:52415 \
   MODEL_ID=kernelpool/MiMo-V2.5-Pro-6bit \
   MIMO_MTP_SIDECAR_PATH=/path/to/model_mtp.safetensors \
   bash scripts/smoke_mimo_mtp_two_node_cluster.sh --live

Pass criteria are intentionally separated in smoke-summary.json:
- runner_result.pass=true: both AR and MTP benchmark runner rows completed as live rows.
- qa_verdict.pass=true: required live telemetry was present and parseable.
- acceptance_result.pass=true: MTP used accepted_execution_path=mimo_mtp_fastpath and mtp_execution_state=successful_mtp.
- throughput_result.pass=true: AR generation_tps > 0 and MTP generation_tps >= MIN_MTP_TOK_S (default 25).
- smoke_result.pass=true only when all of the above are true.
EOF
}

summarize_smoke_result() {
  local log_file="$1"
  local summary_file="$2"
  local min_mtp_tok_s="$3"
  "${PYTHON_CMD[@]}" - "${log_file}" "${summary_file}" "${min_mtp_tok_s}" <<'PY'
import json
import sys
from pathlib import Path

log_file = Path(sys.argv[1])
summary_file = Path(sys.argv[2])
min_mtp_tok_s = float(sys.argv[3])
rows = []
for line in log_file.read_text(encoding='utf-8').splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(row, dict):
        rows.append(row)

metric_rows = [row for row in rows if row.get('evidence_kind') == 'cluster_benchmark_metric']
ar_rows = [row for row in metric_rows if row.get('mode') == 'ar']
mtp_rows = [row for row in metric_rows if row.get('mode') == 'mtp-d1']
latest_ar = ar_rows[-1] if ar_rows else {}
latest_mtp = mtp_rows[-1] if mtp_rows else {}

def as_float(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None

def is_live_row(row):
    return row.get('row_status') == 'live'

ar_tps = as_float(latest_ar.get('generation_tps'))
mtp_tps = as_float(latest_mtp.get('generation_tps'))
mtp_path = latest_mtp.get('accepted_execution_path') or latest_mtp.get('live_execution_path')
mtp_state = latest_mtp.get('mtp_execution_state')
accepted_depth_counts = latest_mtp.get('accepted_depth_counts')
attempted_depth_counts = latest_mtp.get('attempted_depth_counts')
acceptance_rate = latest_mtp.get('acceptance_rate')
sidecar_status = latest_mtp.get('mtp_sidecar_status')
timing_breakdown = latest_mtp.get('timing_breakdown_seconds')

runner_result = {
    'pass': bool(is_live_row(latest_ar) and is_live_row(latest_mtp)),
    'ar_row_status': latest_ar.get('row_status'),
    'mtp_row_status': latest_mtp.get('row_status'),
    'ar_http_status': latest_ar.get('http_status'),
    'mtp_http_status': latest_mtp.get('http_status'),
}
qa_verdict = {
    'pass': bool(
        ar_tps is not None
        and mtp_tps is not None
        and mtp_path is not None
        and mtp_state is not None
        and accepted_depth_counts is not None
        and attempted_depth_counts is not None
        and acceptance_rate is not None
        and sidecar_status is not None
        and timing_breakdown is not None
    ),
    'telemetry_present': {
        'ar_generation_tps': ar_tps is not None,
        'mtp_generation_tps': mtp_tps is not None,
        'accepted_execution_path': mtp_path is not None,
        'mtp_execution_state': mtp_state is not None,
        'accepted_depth_counts': accepted_depth_counts is not None,
        'attempted_depth_counts': attempted_depth_counts is not None,
        'acceptance_rate': acceptance_rate is not None,
        'mtp_sidecar_status': sidecar_status is not None,
        'timing_breakdown_seconds': timing_breakdown is not None,
    },
}
acceptance_result = {
    'pass': bool(mtp_path == 'mimo_mtp_fastpath' and mtp_state == 'successful_mtp'),
    'mtp_accepted_execution_path': mtp_path,
    'mtp_execution_state': mtp_state,
    'accepted_depth_counts': accepted_depth_counts,
    'attempted_depth_counts': attempted_depth_counts,
    'acceptance_rate': acceptance_rate,
}
throughput_result = {
    'pass': bool(ar_tps is not None and ar_tps > 0 and mtp_tps is not None and mtp_tps >= min_mtp_tok_s),
    'ar_generation_tps': ar_tps,
    'mtp_generation_tps': mtp_tps,
    'mtp_min_tok_s': min_mtp_tok_s,
    'target_30_tok_s_met': bool(mtp_tps is not None and mtp_tps >= 30),
    'preferred_40_tok_s_met': bool(mtp_tps is not None and mtp_tps >= 40),
}
smoke_result = {
    'pass': bool(
        runner_result['pass']
        and qa_verdict['pass']
        and acceptance_result['pass']
        and throughput_result['pass']
    ),
    'runner_pass': runner_result['pass'],
    'qa_pass': qa_verdict['pass'],
    'acceptance_pass': acceptance_result['pass'],
    'throughput_pass': throughput_result['pass'],
}
existing = json.loads(summary_file.read_text(encoding='utf-8'))
existing.update({
    'runner_result': runner_result,
    'qa_verdict': qa_verdict,
    'acceptance_result': acceptance_result,
    'throughput_result': throughput_result,
    'smoke_result': smoke_result,
})
summary_file.write_text(json.dumps(existing, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({'smoke_result': smoke_result, 'summary_file': str(summary_file)}, sort_keys=True))
if not smoke_result['pass']:
    print('ERROR: live MTP smoke failed pass criteria; see ' + str(summary_file), file=sys.stderr)
    sys.exit(3)
PY
}

run_local_gate() {
  bash -n "$0"
  "${PYTHON_CMD[@]}" scripts/bench_mimo_mtp_cluster.py --help >/dev/null
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local log_file="${tmp_dir}/synthetic-smoke-output.jsonl"
  local summary_file="${tmp_dir}/synthetic-smoke-summary.json"
  cat >"${summary_file}" <<'JSON'
{"cluster_state":{"two_node_ready":true,"node_count":2,"nodes":["studio1","studio2"],"errors":[]}}
JSON
  cat >"${log_file}" <<'JSONL'
{"evidence_kind":"cluster_benchmark_metric","mode":"ar","row_status":"live","http_status":200,"generation_tps":12.5}
{"evidence_kind":"cluster_benchmark_metric","mode":"mtp-d1","row_status":"live","http_status":200,"generation_tps":31.0,"accepted_execution_path":"mimo_mtp_fastpath","mtp_execution_state":"successful_mtp","accepted_depth_counts":{"1":8},"attempted_depth_counts":{"1":10},"acceptance_rate":0.8,"mtp_sidecar_status":"loaded","timing_breakdown_seconds":{"draft":0.1}}
JSONL
  summarize_smoke_result "${log_file}" "${summary_file}" "25" >/dev/null
  "${PYTHON_CMD[@]}" - "${summary_file}" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding='utf-8'))
assert summary['runner_result']['pass'] is True
assert summary['qa_verdict']['pass'] is True
assert summary['acceptance_result']['pass'] is True
assert summary['throughput_result']['pass'] is True
assert summary['smoke_result']['pass'] is True
PY
  rm -rf "${tmp_dir}"
  printf 'PASS: local two-node smoke harness gate (no network, no live requests)\n' >&2
}

case "${1:-}" in
  --print-checklist)
    print_checklist
    exit 0
    ;;
  --local-gate)
    run_local_gate
    exit 0
    ;;
  --help|-h)
    print_checklist
    cat <<EOF

Environment knobs:
  API_BASE                 default: ${API_BASE}
  MODEL_ID                 default: ${MODEL_ID}
  MIMO_MTP_SIDECAR_PATH    optional official model_mtp.safetensors path
  MAX_TOKENS               default: ${MAX_TOKENS}
  MIN_MTP_TOK_S            default: ${MIN_MTP_TOK_S}
  OUT_DIR                  default: ${OUT_DIR}
  PROMPT                   default: ${PROMPT}

Commands:
  --local-gate             validate harness locally; no network and no live requests
  --print-checklist        print live execution checklist
  --live                   run the explicit live API smoke against API_BASE
EOF
    exit 0
    ;;
  --live)
    ;;
  *)
    print_checklist >&2
    printf '\nERROR: refusing to send live requests without explicit --live. Run --local-gate first, then --live when the cluster is ready.\n' >&2
    exit 64
    ;;
esac

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/smoke-output.jsonl"
SUMMARY_FILE="${OUT_DIR}/smoke-summary.json"

printf 'Writing smoke evidence to %s\n' "${OUT_DIR}" >&2
printf 'Probe API: %s\n' "${API_BASE}" >&2
printf 'Non-destructive live mode: this script will only call /state, /v1/models, and /v1/chat/completions.\n' >&2

"${PYTHON_CMD[@]}" - "${API_BASE}" "${SUMMARY_FILE}" <<'PY'
import datetime as _dt
import json
import sys
import urllib.request

api_base = sys.argv[1].rstrip('/')
summary_file = sys.argv[2]
summary = {
    'api_base': api_base,
    'checked_at_utc': _dt.datetime.now(_dt.UTC).isoformat(),
    'two_node_ready': False,
    'node_count': 0,
    'nodes': [],
    'errors': [],
}
try:
    with urllib.request.urlopen(api_base + '/state', timeout=10) as response:
        state = json.loads(response.read().decode())
    topology = state.get('topology') or {}
    raw_nodes = topology.get('nodes') or topology.get('nodeIds') or state.get('nodes') or []
    if isinstance(raw_nodes, dict):
        nodes = sorted(str(key) for key in raw_nodes)
    elif isinstance(raw_nodes, list):
        nodes = sorted(str(item.get('id') or item.get('node_id') or item) if isinstance(item, dict) else str(item) for item in raw_nodes)
    else:
        nodes = []
    last_seen = state.get('lastSeen') or state.get('last_seen') or {}
    if not nodes and isinstance(last_seen, dict):
        nodes = sorted(str(key) for key in last_seen)
    summary.update({
        'node_count': len(nodes),
        'nodes': nodes,
        'last_seen_keys': sorted(str(key) for key in last_seen) if isinstance(last_seen, dict) else [],
        'two_node_ready': len(nodes) >= 2,
    })
except Exception as exc:  # noqa: BLE001 - smoke script reports concrete blocker
    summary['errors'].append(f'/state probe failed: {exc}')

with open(summary_file, 'w', encoding='utf-8') as file:
    json.dump({'cluster_state': summary}, file, indent=2, sort_keys=True)
    file.write('\n')
print(json.dumps({'cluster_state': summary}, sort_keys=True))
if not summary['two_node_ready']:
    print('ERROR: /state does not show at least two cluster nodes; refusing throughput smoke.', file=sys.stderr)
    sys.exit(2)
PY

payload='{"mimo_mtp_fastpath":true,"mimo_mtp_depth":1,"mimo_mtp_fail_closed":true}'
if [[ -n "${SIDECAR_PATH}" ]]; then
  payload=$("${PYTHON_CMD[@]}" - "${SIDECAR_PATH}" <<'PY'
import json, sys
print(json.dumps({
    'mimo_mtp_fastpath': True,
    'mimo_mtp_depth': 1,
    'mimo_mtp_fail_closed': True,
    'mimo_mtp_sidecar_path': sys.argv[1],
}, separators=(',', ':'), sort_keys=True))
PY
)
fi

{
  "${PYTHON_CMD[@]}" scripts/bench_mimo_mtp_cluster.py \
    --api-base "${API_BASE}" \
    --model-id "${MODEL_ID}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_TOKENS}" \
    --repeats 1 \
    --mode-label ar \
    --list-models \
    --output-dir "${OUT_DIR}"

  "${PYTHON_CMD[@]}" scripts/bench_mimo_mtp_cluster.py \
    --api-base "${API_BASE}" \
    --model-id "${MODEL_ID}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_TOKENS}" \
    --repeats 1 \
    --mode-label mtp-d1 \
    --payload-extra-json "${payload}" \
    --output-dir "${OUT_DIR}"
} | tee "${LOG_FILE}"

summarize_smoke_result "${LOG_FILE}" "${SUMMARY_FILE}" "${MIN_MTP_TOK_S}"

printf 'PASS: live two-node MiMo MTP smoke evidence: %s\n' "${SUMMARY_FILE}" >&2
