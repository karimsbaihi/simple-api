#!/bin/bash
set -o errexit
set -o xtrace

echo "=== Starting Build Process ==="
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

echo "=== Installing Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build Completed Successfully ==="