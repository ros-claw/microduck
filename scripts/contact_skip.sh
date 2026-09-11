#!/usr/bin/env bash
# Physical contact replay. Extra arguments override these defaults.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec .venv/bin/python scripts/eval_physical_skip.py \
  --hop policies/ropehop_contact.onnx --seconds 30 --seed 0 \
  --rope-joint-type ball --physics-dt .0002 --rope-radius .0015 \
  --rope-floor-timeconst .0004 --hop-start-delay 0 --settle-seconds 0 \
  --geometric-timing --max-turn-hz 3.1 --rope-length .58 --jumper-y 0 \
  --rope-initial-phase 1.57079632679 --rope-velocity-limit 0 \
  --output artifacts/takeover/contact-latest.json "$@"
