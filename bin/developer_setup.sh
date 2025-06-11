#!/bin/bash
set -e

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install it first."
    echo "See: https://github.com/astral-sh/uv"
    exit
fi

# Create virtual environment with a specific python version
uv venv .venv -p python3.11
source .venv/bin/activate

# Check for macOS on ARM
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "macOS on ARM detected. Installing decord using the recommended method..."
    uv pip install eva-decord

    echo "Installing other dependencies from requirements.txt..."
    grep -v '^decord' requirements.txt > requirements.tmp
    uv pip install -r requirements.tmp --prerelease=allow
    rm requirements.tmp
else
    echo "Installing all dependencies from requirements.txt..."
    uv pip install -r requirements.txt --prerelease=allow
fi

echo "Developer setup complete. Virtual environment created at ./.venv with Python 3.11."
echo "To activate it, run: source .venv/bin/activate" 