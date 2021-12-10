#!/bin/bash

# Build library
python setup.py bdist_wheel

# Install library
pip install dist/wavy*

# Remove folders
rm -rf build
rm -rf dist