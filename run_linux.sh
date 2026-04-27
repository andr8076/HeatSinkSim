#!/usr/bin/env bash
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 run_thermal_sim.py
else
  echo "python3 not found. Install Python 3 with your package manager."
  read -r -p "Press Enter to close..."
fi
