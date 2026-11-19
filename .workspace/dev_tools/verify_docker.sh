#!/usr/bin/env bash
# Offline container smoke test: describe + streaming predict, no network, non-root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-rsi-submission:1.0}"
ARTIFACT="${1:-/tmp/local_host_a/artifact_01}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CTX='{"protocol":"contest.predict-context.v1","dataset_key":"A","task":"classification","artifact":{},"artifact_id":"unit"}'
python3 - "$WORK/input.jsonl" <<'PY'
import json, sys
rows = [{"task": "classification", "inputs": {"features": [0.1] * 128}},
        {"task": "classification", "inputs": {"features": [0.2] * 128}}]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
PY
echo '=== image size ==='
docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}} {{.Size}}'
echo '=== describe (--network none) ==='
docker run --rm --network none "$IMAGE" python3 /app/harness.py describe
echo '=== predict (--network none, read-only model, uid 65534) ==='
docker run --rm --network none --user 65534:65534 \
  -e CONTEST_PREDICTION_CONTEXT="$CTX" \
  -e CONTEST_DATASET_KEY=A -e CONTEST_TASK=classification \
  -e RSI_MODEL_ROOT=/workspace/model \
  -e CONTEST_ARTIFACT_PATH=/workspace/model/checkpoint.npz \
  -v "$ARTIFACT:/workspace/model:ro" \
  -i "$IMAGE" python3 /app/harness.py predict < "$WORK/input.jsonl"
