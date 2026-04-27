#!/bin/zsh
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
read -k 1 "?Press any key to close..."
