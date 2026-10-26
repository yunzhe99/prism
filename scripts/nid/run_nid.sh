#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 EMBEDDINGS.npy LABELS.npy VARIANT [additional nid.py arguments]" >&2
  echo "VARIANT: swap, legacy_random, or legacy_swap" >&2
  exit 2
fi

NID_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NID_REPO_ROOT="$(cd "${NID_SCRIPT_DIR}/../.." && pwd)"
NID_EMBEDDINGS="$1"
NID_LABELS="$2"
NID_VARIANT="$3"
shift 3
python3 "${NID_REPO_ROOT}/nid.py" \
  --embeddings "${NID_EMBEDDINGS}" --labels "${NID_LABELS}" \
  --variant "${NID_VARIANT}" "$@"
