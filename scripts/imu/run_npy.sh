#!/usr/bin/env bash
# Explicit example interface for any of uci/hhar/motion/shl.
# It does not choose task labels, normalization, domains, or paper settings.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$repo_root/prism.py" "$@"
