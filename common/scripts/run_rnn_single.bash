#!/usr/bin/env bash

echo $1
export TOKENIZATION_BENCHMARK=$1
python3.5 run_train.py -xvt
python3.5 run_benchmarks.py -xvc
echo -e "\n"
