#!/usr/bin/env bash
# End-to-end demo of the koharu-rpc script surface: start the server, open a
# project, add a page, run OCR and translation over it, poll until the run
# finishes, and export the rendered page.
#
# Documented, NOT executed by CI: the pipeline stages download model weights
# on first use, and repo AGENTS.md runs e2e only on explicit request.
# To run it by hand:
#   1. Prefetch the weights once with the headless pipeline binary:
#        cargo run -p koharu-pipeline --bin run -- --cpu \
#          --input page.png --output out.png
#      Models cache under ~/.cache/huggingface; later runs need no network.
#   2. bash scripts/e2e-rpc-demo.sh
#
# Plain bash + curl; no jq, no test framework.
set -euo pipefail

PORT="${PORT:-4100}"
BASE="http://127.0.0.1:${PORT}/api/v1"
WORK="$(mktemp -d /tmp/koharu-e2e.XXXXXX)"
trap 'kill "${SERVER:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# 8x8 synthetic page; the flow is what is being proven, not model quality.
printf '%s' 'iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAXklEQVR4nBXKMQEEMQgAsIqoCEQgAhGIYDwBeP5Ph2w553y/S5AUzbCccwWCpGiGvS+EQJAUzbDxQgoESdEMmy+UQJAUzbD1QgsESdEM2y+MQJAUzbDzwgoESdEMyx9X2YLBG/IYQwAAAABJRU5ErkJggg==' |
	base64 -d >"$WORK/page.png"

cargo build -p koharu-rpc
./target/debug/koharu-rpc --port "$PORT" &
SERVER=$!

for _ in $(seq 1 120); do
	curl -fsS "$BASE/health" >/dev/null 2>&1 && break
	sleep 1
done
curl -fsS "$BASE/health" >/dev/null

curl -fsS -X POST "$BASE/project/open" \
	-H 'content-type: application/json' \
	-d "{\"path\": \"$WORK/project\", \"create\": true}" >/dev/null

curl -fsS -X POST "$BASE/project/pages" \
	-H 'content-type: application/json' \
	-d "{\"paths\": [\"$WORK/page.png\"]}" >/dev/null

RUN=$(curl -fsS -X POST "$BASE/pipelines" \
	-H 'content-type: application/json' \
	-d '{"stages": ["ocr", "translation"]}')
ID=$(sed -n 's/.*"operationId":"\([^"]*\)".*/\1/p' <<<"$RUN")

STATUS=""
for _ in $(seq 1 300); do
	JOBS=$(curl -fsS "$BASE/operations")
	STATUS=$(sed -n "s/.*\"id\":\"$ID\",\"status\":\"\([a-z_]*\)\".*/\1/p" <<<"$JOBS")
	case "$STATUS" in
	completed | stopped | failed) break ;;
	esac
	sleep 2
done

if [ "$STATUS" != completed ]; then
	echo "pipeline run ended with status: $STATUS" >&2
	exit 1
fi

EXPORT=$(curl -fsS -X POST "$BASE/export" \
	-H 'content-type: application/json' \
	-d "{\"dir\": \"$WORK/export\"}")
FILE=$(sed -n 's/.*"files":\["\([^"]*\)".*/\1/p' <<<"$EXPORT")
test -s "$FILE"
echo "exported $FILE"
