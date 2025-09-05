#!/bin/bash
set -o errexit
set -o xtrace

echo "=== Starting Build Process ==="
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

echo "=== Installing Git LFS ==="
apt-get update
apt-get install -y git-lfs

echo "=== Setting up Git LFS ==="
git lfs install
git lfs version

echo "=== Pulling LFS Files ==="
git lfs pull --include="best_model.pt,class_mappings.pkl" || echo "Git LFS pull may have failed, continuing..."

echo "=== Verifying Files ==="
echo "Files after Git LFS pull:"
ls -la

# Check if files exist and their sizes
if [ -f "best_model.pt" ]; then
    echo "Model file size: $(du -h best_model.pt)"
    echo "Model file type: $(file -b best_model.pt)"
else
    echo "ERROR: Model file not found!"
    exit 1
fi

if [ -f "class_mappings.pkl" ]; then
    echo "Mappings file size: $(du -h class_mappings.pkl)"
else
    echo "ERROR: Mappings file not found!"
    exit 1
fi

echo "=== Installing Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build Completed Successfully ==="