#!/bin/bash

# Clone the repository (delete existing if necessary)
REPO_URL="https://github.com/abrarsheikh/blog.git"
REPO_DIR="blog"

if [ -d "$REPO_DIR" ]; then
    rm -rf "$REPO_DIR"
fi

git clone "$REPO_URL"

# Change into the project directory
cd "$REPO_DIR" || exit

pip install --upgrade pip
pip install poetry

# Install the dependencies
poetry install
