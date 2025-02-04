#!/bin/bash

# Check if exactly one argument (the notebook path) is provided.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_notebook_file.ipynb>"
  exit 1
fi

NOTEBOOK_FILE="$1"

# Verify that the input file exists.
if [ ! -f "$NOTEBOOK_FILE" ]; then
  echo "Error: File '$NOTEBOOK_FILE' not found."
  exit 1
fi

# Ensure the file has a .ipynb extension.
if [[ "$NOTEBOOK_FILE" != *.ipynb ]]; then
  echo "Error: Input file must be a .ipynb notebook."
  exit 1
fi

# Get the directory and base name of the notebook file.
OUTPUT_DIR=$(dirname "$NOTEBOOK_FILE")
BASE_NAME=$(basename "$NOTEBOOK_FILE" .ipynb)
PY_FILE="${OUTPUT_DIR}/${BASE_NAME}.py"

# Convert the notebook to a Python script, placing the output in the same directory.
jupyter nbconvert --to script --output-dir "$OUTPUT_DIR" "$NOTEBOOK_FILE"

if [ $? -ne 0 ]; then
  echo "Conversion failed."
  exit 1
fi

echo "Conversion successful: ${PY_FILE} created."

# Check if the converted Python file exists.
if [ ! -f "$PY_FILE" ]; then
  echo "Error: Converted file '${PY_FILE}' not found."
  exit 1
fi

# Run the Python script.
echo "Running ${PY_FILE}..."
python3 "$PY_FILE"
