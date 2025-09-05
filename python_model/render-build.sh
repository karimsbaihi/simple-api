#!/bin/bash
set -o errexit
set -o xtrace

echo "=== Starting Build Process ==="
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

echo "=== Checking Files ==="
# Verify files exist
if [ ! -f "optimized_model.pt" ]; then
    echo "ERROR: optimized_model.pt not found!"
    exit 1
fi

if [ ! -f "class_mappings.pkl" ]; then
    echo "ERROR: class_mappings.pkl not found!"
    exit 1
fi

echo "Model file size: $(du -h optimized_model.pt)"
echo "Mappings file size: $(du -h class_mappings.pkl)"

echo "=== Installing Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build Completed Successfully ==="