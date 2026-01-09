#!/bin/bash
echo "Starting all experiments..."
echo

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

for file in journal_paper_ran_experiments/*.py; do
    echo "Running $file..."
    python "$file"
    echo "Completed $file"
    echo
done

echo "All experiments completed!"