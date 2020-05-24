#!/usr/bin/env python3
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from constants import DEFAULT_BENCHMARK_DUMP_DIR


def read(fl):
    with open(fl, 'r') as fil:
        res = fil.read()
    assert res.count('\n') < 2, fl
    return res.replace('\n', '')


if __name__ == "__main__":
    for benchmark in filter(lambda x: os.path.isdir(os.path.join(DEFAULT_BENCHMARK_DUMP_DIR,  x)), os.listdir(DEFAULT_BENCHMARK_DUMP_DIR)):
        root_path = os.path.join(DEFAULT_BENCHMARK_DUMP_DIR, benchmark, 'fixed')
        output_path = os.path.join(DEFAULT_BENCHMARK_DUMP_DIR, benchmark, 'all_fixed.txt')
        output_path2 = os.path.join('results', benchmark + '_all_fixed.txt')
        all_texts = sorted([os.path.join(root_path, fl)
                            for fl in os.listdir(root_path)
                            if fl.endswith('.txt')])

        if os.path.isfile(output_path2):
            print('skipping..', output_path2)
        else:
            res = "\n".join((read(fl) for fl in tqdm(all_texts)))
            with open(output_path, 'w') as fl:
                fl.write(res)
            print(output_path, 'written..')

            with open(output_path2, 'w') as fl:
                fl.write(res)
            print(output_path2, 'written..')
