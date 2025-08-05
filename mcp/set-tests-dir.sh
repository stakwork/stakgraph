#!/bin/bash
if [ -z "$1" ]; then
    export TESTS_DIR="$(dirname "$0")/tests/generated_tests"
else
    export TESTS_DIR="$1"
fi
echo "Tests directory set to: $TESTS_DIR"