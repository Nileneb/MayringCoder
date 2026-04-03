#!/usr/bin/env bash
set -euo pipefail


create () {
  local file="$1"
  echo "Creating agent from $file"
  curl -sS -X POST "https://api.langdock.com/agent/v1/create" \
    -H "Authorization: Bearer $MASTER_LANGDOCK" \
    -H "Content-Type: application/json" \
    --data @"$file"
  echo
}

create agent_pico.json
create agent_mapping.json
create agent_retrieval.json
create agent_screening.json
create agent_synthesis.json
create agent_mayring.json