#!/bin/bash
echo "Starting all experiments..."
echo

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

for file in journal_paper_experiments_with_algorithm/*.py; do
    echo "Running $file..."
    python "$file"
    echo "Completed $file"
    echo
done

echo "All experiments completed!"