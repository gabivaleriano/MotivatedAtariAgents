#!/usr/bin/env bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running script..."
python main.py --mode full --save-dir results
