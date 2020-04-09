#!/usr/bin/env bash

for x in {0..32}
do
    export TOKENIZATION_BENCHMARK=$x
    python3 run_benchmarks.py -xvd
    echo -e "\n"
done
