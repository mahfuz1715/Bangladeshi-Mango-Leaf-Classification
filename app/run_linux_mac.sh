#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -f ".venv/bin/python" ]; then
  .venv/bin/python -m streamlit run app.py
else
  python3 -m streamlit run app.py
fi
