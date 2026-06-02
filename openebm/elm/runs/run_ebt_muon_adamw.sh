#!/bin/bash

# Compatibility entrypoint for the current EBT Muon+AdamW SFT script.
# Keep the historical path name, but route new train-engine testing through the
# context-1024 branch. Exact second-order EBT training is activation-heavy, and
# the context-2048 branch can OOM even under ZeRO-2.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_ebt_muon_adamw_c1024.sh" "$@"
