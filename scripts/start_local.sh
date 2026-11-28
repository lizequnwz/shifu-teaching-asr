#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi

uv pip install --python .venv/bin/python -r requirements-workflow.txt -r requirements-web.txt
npm --prefix web ci
npm --prefix web run build

exec .venv/bin/python scripts/subtitle_web.py
