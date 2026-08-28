#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment using uv..."
    if command -v uv &> /dev/null; then
        uv venv --python 3.11 .venv
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt
    fi
fi

# Run Streamlit App
echo "Starting Gene Disease Diagnosis Streamlit App..."
.venv/bin/streamlit run app.py
