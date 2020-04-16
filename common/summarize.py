#!/usr/bin/env python3
import os

import pandas as pd
import numpy as np

from constants import DEFAULT_BENCHMARK_DUMP_DIR


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall


def fscore(p, r):
    return 2 * p * r / (p + r + 1e-8)


def benchmark_name(name):
    return name
    return '_'.join(name.split('-')[1].split('_')[:2])


def summarize(benchmark, results_path):
    #,precision,recall,f1score,precision to add,recall to add,f1score to add,
    #precision to del,recall to del,f1score to del,tp,fp,fn,tp to add,
    #fp to add,fn to add,tp to del,fp to del,fn to del,acc,duration
    try:
        df = pd.read_csv(results_path)
    except Exception as err:
        print("can't parse", results_path)
        return

    tp = df['tp'].values
    fp = df['fp'].values
    fn = df['fn'].values
    accuracy = df['acc'].values.mean()
    duration = df['duration'].values.mean()

    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    precision_micro, recall_micro = precision_recall(tp_micro, fp_micro, fn_micro)
    fscore_micro = fscore(precision_micro, recall_micro)

    mask = (tp == 0) & (fp == 0) & (fn == 0)
    tp[mask] = 1

    precision_macro, recall_macro = precision_recall(tp, fp, fn)
    precision_macro = precision_macro.mean()
    recall_macro = recall_macro.mean()
    fscore_macro = fscore(precision_macro, recall_macro)

    print("%s : %d sentences\n"
          "macro:\tP: %.5f\tR: %.5f\tF: %.5f\n"
          "micro:\tP: %.5f\tR: %.5f\tF: %.5f\n"
          "duration: %.2f s/seq\n"
          "Seq acc: %.5f\n------" % (
              benchmark_name(benchmark), df.shape[0],
              precision_macro, recall_macro, fscore_macro,
              precision_micro, recall_micro, fscore_micro,
              duration, accuracy))


if __name__ == '__main__':
    for benchmark in sorted(os.listdir(DEFAULT_BENCHMARK_DUMP_DIR)):
        results_path = os.path.join(DEFAULT_BENCHMARK_DUMP_DIR, benchmark, 'results.csv')
        if os.path.isfile(results_path):
            summarize(benchmark, results_path)
