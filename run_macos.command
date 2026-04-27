#!/bin/zsh
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 run_thermal_sim.py
else
  echo "python3 not found. Install Python 3 from https://www.python.org/downloads/macos/ or via Homebrew."
  read -k 1 "?Press any key to close..."
fi
