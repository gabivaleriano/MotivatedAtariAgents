#!/usr/bin/env bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running script..."
python main.py --save-dir results
