"""
Module containing running Benchmark object.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import csv
import datetime
import multiprocessing
import re
import os
import warnings

import pandas as pd
import numpy as np

from models import construct_and_load_fixer
from constants import HTML, NUM_THREADS, SMOOTHING, TERMINAL
from handlers.reader import Reader
from utils.logger import logger
from utils.multiviewer import MultiViewer
from utils.utils import (
    chunker, cleanstr, compute_relative_log_improvement,
    extract_file_name, gen_chunker, get_all_files, open_or_create_write_file,
    take_first_n)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")


class Benchmark:
    """
    Benchmarks evaluator of fixers.
    """
    def __init__(self, config):
        self.config = config
        self.random_sample_files = config.random_sample_files
        self.dump_dir = config.dump_dir
        self.construct_timestamp = datetime.datetime.now()
        self.fixer_repr = config.fixer_repr
        self.reader = Reader(config)
        self.metrics = ['precision', 'recall', 'f1score',
                        'precision to add', 'recall to add', 'f1score to add',
                        'precision to del', 'recall to del', 'f1score to del',
                        'tp', 'fp', 'fn', 'tp to add', 'fp to add',
                        'fn to add', 'tp to del', 'fp to del', 'fn to del',
                        'acc', 'duration']

    def __repr__(self):
        return self.fixer_repr

    def get_timestamp_folder_name(self):
        """
        Get the timestamp of the starting time of the Benchmarks.

        :returns: Folder with timestamp string
        """
        tim = self.construct_timestamp
        return '%s-D%.2d-%.2d_T%.2d-%.2d-%.2d' % (self.dump_dir, tim.day,
                                                  tim.month, tim.hour,
                                                  tim.minute, tim.second)

    def run_benchmark(self, key):
        """
        Run the benchmarks of a given file or directory.

        :param str path: Correct file path
        :rtype: dict
        :returns:
            Evaluation dictionary of the metric evaluations of the fixed file.
        """
        fixer = construct_and_load_fixer(self.config)
        file_name, correct_text, corrupt_text = key
        correct_text = re.sub(r' +', ' ', correct_text).strip()
        corrupt_text = re.sub(r' +', ' ', corrupt_text).strip()
        corrupt_path = file_name

        row = {}
        comparisons = []
        # html_comparisons = []
        evaluator = MultiViewer()
        if len(correct_text) >= 12000:
            logger.log_info("%s is too big, won't fix.." % corrupt_path)
            row[file_name] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            return row

        # Fixing
        logger.log_info("with %s Fixing.. " % str(fixer) + corrupt_path + '\n')
        t0 = datetime.datetime.now()
        fixed_text = fixer.fix(corrupt_text)
        duration = datetime.datetime.now() - t0
        caption = 'fixing %s with %s' % (file_name, str(fixer))

        # comparing
        # metrics_vals, comparison, html_comparison = evaluator.evaluate(
        #     correct_text, corrupt_text, fixed_text,
        #     modes=[TERMINAL, HTML], caption=caption)
        metrics_vals, comparison = evaluator.evaluate(
            correct_text, corrupt_text, fixed_text,
            modes=TERMINAL, caption=caption)
        row[file_name] = metrics_vals + (duration.total_seconds(),)
        comparisons.append(comparison)
        # html_comparisons.append(html_comparison)

        # Fixed file
        logger.log_info('with %s Fixed.. ' % str(fixer) + corrupt_path + '\n',
                        comparison, '\nFixing had a duration of',
                        int(round(duration.total_seconds())), 's')
        fixed_path = os.path.join(self.get_timestamp_folder_name(),
                                  'fixed', file_name + '_fixed.txt')
        with open_or_create_write_file(fixed_path) as fixed_file:
            fixed_file.write(fixed_text)
            fixed_file.close()
            logger.log_report('dumped fixed file into:', fixed_path)

        # metric results
        metric_comparison = evaluator.metric_comparison(
            row, self.metrics, modes=TERMINAL)
        # metric_comparison,html_metric_comparison= evaluator.metric_comparison(
        #     row, self.metrics, modes=[TERMINAL, HTML])
        comparisons.append(metric_comparison)
        # html_comparisons.append(html_metric_comparison)
        logger.log_report(metric_comparison)

        # Terminal dumps
        comparisons = evaluator.merge_wrapped_pages(*comparisons,
                                                    mode=TERMINAL)
        comparison_path = os.path.join(self.get_timestamp_folder_name(),
                                       'comparisons', file_name + '.fix')
        with open_or_create_write_file(comparison_path, 'w') as output:
            output.write(cleanstr('Results of fixing: %s' % corrupt_path))
            output.write(cleanstr(comparisons))
            output.close()
            logger.log_report('dumped comparison into:', comparison_path)

        # HTML dumps
        # html_comparisons = evaluator.merge_wrapped_pages(*html_comparisons,
        #                                                  mode=HTML)
        # html_comparison_path = os.path.join(self.get_timestamp_folder_name(),
        #                                     'comparisons', 'htmls',
        #                                     file_name + '.html')
        # with open_or_create_write_file(html_comparison_path, 'w') as output:
        #     output.write(cleanstr(html_comparisons))
        #     output.close()
        #     logger.log_report('dumped html comparison into:',
        #                       html_comparison_path)
        return row

    def update_csv(self, rows, csv_path, files, first_row=False):
        """
        Dump the results in a csv file.

        :param dict rows:
            Evaluation dictionary of the metric evaluations of the fixed files
        :param str csv_path: Output csv file path
        :param list(str) files: names of the files
        :param bool first_row: Flag declaring if the first row being written
        """
        logger.output(rows)
        keys = sorted(rows[1].keys())

        with open(csv_path, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if first_row:
                row = ['Files', '', 'Improvement']
                for key in keys:
                    row += [key, 'precision', 'recall', 'f1score',
                            'precision to add', 'recall to add',
                            'f1score to add', 'precision to del',
                            'recall to del', 'f1score to del',
                            'tp', 'fp', 'fn', 'tp to add', 'fp to add',
                            'fn to add', 'tp to del', 'fp to del', 'fn to del']
                writer.writerow(list(map(cleanstr, row)))

            for path, row in zip(files, rows):

                pt, file_name, file_ext = extract_file_name(path)
                row_out = [file_name.replace(',', '.') + '.' + file_ext,
                           '', '']
                f_scores = []
                for metrics_vals in (row[key] for key in keys if row):
                    row_out += [''] + ['%.5f' % val for val in metrics_vals]
                    f_scores.append(metrics_vals[2])
                if f_scores and len(f_scores) >= 2:
                    row_out[2] = '%.5f' % compute_relative_log_improvement(
                        f_scores[1], f_scores[0])
                writer.writerow(list(map(cleanstr, row_out)))

    def summarize(self, mean_dict):
        """
        Summarize the evaluated results.

        :param dict mean_dict:
            Evaluation dictionary of the metric evaluations of the fixed files
        """
        viewer = MultiViewer()
        logger.log_report(viewer.metric_comparison(mean_dict, self.metrics))
        # self.summarize_html(mean_dict, viewer)

    def summarize_html(self, mean_dict, viewer):
        html_comparison = viewer.metric_comparison(mean_dict, self.metrics,
                                                   modes=HTML)
        result = [html_comparison]
        path = os.path.join(self.get_timestamp_folder_name(),
                            'comparisons', 'htmls')
        result.extend((open(fil, 'r').read()
                       for fil in get_all_files(path, ext='.html')
                       if fil != 'summary.html'))
        path = os.path.join(path, 'summary.html')
        with open(path, 'w') as summary_file:
            summary_file.write(viewer.merge_wrapped_pages(*result, mode=HTML,
                                                          seperate=True))
            summary_file.close()
            logger.log_report('summarized files in %s' % path)
        # return result

    def get_scores(self, df):

        def prf(tp, fp, fn):
            precision = tp / float(tp + fp + SMOOTHING)
            recall = tp / float(tp + fn + SMOOTHING)
            fscore = 2 * recall * precision / (recall + precision + SMOOTHING)
            if tp == fp == fn == 0:
                precision = recall = fscore = 1.0
            return precision, recall, fscore

        sums = df.sum(axis=0)
        tp, fp, fn, tp_add, fp_add, fn_add, tp_del, fp_del, fn_del =\
            sums.values[9:18]
        precision, recall, fscore = prf(tp, fp, fn)
        precision_add, recall_add, fscore_add = prf(tp_add, fp_add, fn_add)
        precision_del, recall_del, fscore_del = prf(tp_del, fp_del, fn_del)
        return (precision, recall, fscore, precision_add, recall_add,
                fscore_add, precision_del, recall_del, fscore_del)

    def run(self):
        """
        Run the benchmark on all the files.
        """
        logger.log_report('starting', self.fixer_repr)
        score_rows = {}
        mean_dict = {}
        files = self.reader.read_test_pairs()
        if self.random_sample_files:
            files = take_first_n(files, 10)

        logger.start()
        num_files = 0
        csv_path = os.path.join(self.get_timestamp_folder_name(),
                                'results.csv')
        open_or_create_write_file(csv_path)
        for chunk_id, files_collection in enumerate(gen_chunker(files, 200)):
            logger.start()
            # rows = list(map(self.run_benchmark, files_collection))
            with multiprocessing.Pool(NUM_THREADS) as pool:
                rows = list(pool.map(self.run_benchmark, files_collection))
                pool.close()
                pool.join()
                del pool
            num_files += len(rows)
            for row in rows:
                for fil, values in row.items():
                    score_rows[fil] = values
            # self.update_csv(rows, csv_path, files_collection,
            #                 first_row=(chunk_id == 0))
            df = pd.DataFrame.from_dict(score_rows, columns=self.metrics, orient='index')
            df.to_csv(csv_path)

            micro_scores = self.get_scores(df)
            mean_dict[' %d files, macro' % num_files] = np.array(
                list(score_rows.values())).mean(axis=0)
            mean_dict[' %d files, micro' % num_files] = np.array(micro_scores)

            logger.log_report('%d files, seq accuracy %.5f, avg. duration %.2f' % (
                num_files, df['acc'].mean(), df['duration'].mean()))
            self.summarize(mean_dict)

            logger.log_full_report_into_file(os.path.join(
                self.get_timestamp_folder_name(),
                'chunk%d_' % (chunk_id + 1)), keep_log=True)
            logger.log_full_report_into_file('%s-chunk%d_' % (
                self.dump_dir, chunk_id + 1))

        logger.log_seperator()
        mean_dict[' %d files' % num_files] = np.array(
            list(score_rows.values())).mean(axis=0)
        self.summarize(mean_dict)
        logger.log_full_report_into_file(os.path.join(
            self.get_timestamp_folder_name(),
            'all_'), keep_log=True)
        logger.log_full_report_into_file('%s-all_' % self.dump_dir)
