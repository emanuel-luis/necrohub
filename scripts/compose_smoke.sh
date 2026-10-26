#!/usr/bin/env bash
# Checks the running compose stack (db, loader, api, ui). Used by `make smoke` and CI's compose-smoke job.
set -euo pipefail

API="${API_URL:-http://127.0.0.1:8000}"
UI="${UI_URL:-http://127.0.0.1:8501}"
CASE="${SMOKE_CASE:-c7-00003}"

health=$(curl -fsS "$API/healthz")
echo "healthz: $health"
echo "$health" | jq -e '.status == "ok" and .n_chunks > 0' >/dev/null

reply=$(curl -fsS -X POST "$API/ask" -H 'Content-Type: application/json' \
  -d "{\"question\": \"Qual a causa da morte no caso $CASE?\", \"k\": 4}")
echo "$reply" | jq -e --arg case "$CASE" \
  '(.answer | length) > 0 and (.citations | length) > 0 and any(.citations[]; .doc_id == $case)' \
  >/dev/null

ui_health=$(curl -fsS "$UI/_stcore/health")
[ "$ui_health" = "ok" ]

loader_exit=$(docker compose ps -a --format '{{.ExitCode}}' loader)
[ "$loader_exit" = "0" ]
echo "smoke ok"
