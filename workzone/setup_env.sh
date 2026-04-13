#!/bin/bash
# Setup Python environment for Physio_Data on a remote server.
#
# Usage:
#   cd /path/to/Physio_Data
#   bash workzone/setup_env.sh
#
# Creates a conda env named 'physio_data' with all dependencies.

ENV_NAME="physio_data"

echo "=== Physio_Data Environment Setup ==="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install miniconda first."
    exit 1
fi

# Create env if not exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    CONDA_CMD="install"
else
    echo "Creating environment '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.10 -y
    CONDA_CMD="install"
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Python: $(python --version)"
echo "Location: $(which python)"

# Core (always needed)
pip install numpy scipy wfdb

# Preprocessing
pip install polars pyarrow pandas

# Config parsing
pip install pyyaml

# Visualization (for Step 0c demo alignment plots)
pip install matplotlib

# Install physio_data package in editable mode
pip install -e .

echo ""
echo "=== Done ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Next: python workzone/mimic3/step0_explore.py"
