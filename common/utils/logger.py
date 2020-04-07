"""
Module containing class for different logging functionalities.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import datetime
import inspect
import os
import sys
# from pympler import tracker, summary, classtracker
# import pdb

from constants import DEBUG, FULL_DEBUG, VERBOSE
from utils.utils import cleanstr, extract_file_name, open_or_create_write_file

import psutil


class Logger:
    """
    Logger class that print content with different flags
    """
    def __init__(self):
        self.debug = DEBUG
        self.full_debug = FULL_DEBUG
        self.verbose = VERBOSE
        self.construct_timestamp = datetime.datetime.now()
        self.stored_buffer = []
        self.threads_log = ['']
        self.tracked_objects = {}
        self.view_only = None

    def set_view_only(self, highlights=None):
        self.view_only = highlights

    def track_memory(self, msg, highlight=0):
        tot_mem = psutil.Process(os.getpid()).memory_info().rss
        self._log_tag(msg, '%.3f MB' % (tot_mem / 1e6),
                      tot_mem, 'Bytes', tag='DEBUG', errorstream=True,
                      report_log_flag=self.debug, print_flag=self.debug,
                      highlight=highlight)

    def set_debug(self, debug):
        """
        Set the debug flag.

        :param boolean debug: The debug flag
        """
        self.debug = debug

    def set_full_debug(self, full_debug):
        """
        Set the full-debug flag.

        :param boolean full_debug: The full-debug flag
        """
        self.full_debug = full_debug

    def set_verbose(self, verbose):
        """
        Set the verbose flag.

        :param boolean verbose: The verbose flag
        """
        self.verbose = verbose

    def output(self, *msgs):
        """
        output a list of messages

        :param iterable msgs: Arguments to be printed.
        """
        output = ' '.join(map(str, msgs))

        for idx in range(len(self.stored_buffer)):
            self.stored_buffer[idx] += output + '\n'
        print(output)

    def _get_calling_class_name(self, frame):
        if 'self' in frame.f_locals:
            return frame.f_locals['self'].__class__.__name__
        else:
            if '__file__' in frame.f_locals:
                fil = frame.f_locals['__file__']
                return fil.split('/')[-1].split('.py')[0]
            elif hasattr(frame, 'f_back'):
                ret = frame.f_code.co_name
                if len(ret) > 1 and ret[0] == '<' and ret[-1] == '>':
                    ret = ret[1:-1].upper()
                return ret
        return ''

    def _get_time_frame(self):
        prev_frame = self._get_frame().f_back
        frame = prev_frame.f_back
        calling_class = self._get_calling_class_name(frame)
        cwd = os.getcwd()
        # if cwd not in frame.f_code.co_filename:
        #     frame = prev_frame

        file_path = frame.f_code.co_filename.split(cwd)
        if len(file_path) > 1:
            file_path = file_path[1]
        else:
            file_path = frame.f_code.co_filename or '//'
        if len(file_path) > 1 and file_path[0] == '<' and file_path[-1] == '>':
            file_path = file_path[1:-1].upper()
        now = datetime.datetime.now()
        return '[%s:%s:%d][%.2d-%.2d %.2d:%.2d:%.2d]' % (
            file_path,
            calling_class,
            frame.f_lineno, now.day, now.month,
            now.hour, now.minute, now.second)

    def _log_tag(self, *msgs, tag='', errorstream=False, print_flag=True,
                 report_log_flag=False, highlight=0):
        if report_log_flag or print_flag:
            if tag:
                output = ' '.join((self._get_time_frame(), tag + ':',
                                   str.join(' ', map(str, msgs))))
            else:
                output = ' '.join((self._get_time_frame(),
                                   str.join(' ', map(str, msgs))))

            if report_log_flag:
                for idx in range(len(self.stored_buffer)):
                    self.stored_buffer[idx] += output + '\n'
            if print_flag:
                if self.view_only is None or highlight in self.view_only:
                    if highlight:
                        output = "\033[%dm%s\033[0m" % (highlight + 30, output)
                    if errorstream:
                        print(output, file=sys.stderr)
                    else:
                        print(output)

    def log_error(self, *msgs, highlight=0):
        """
        Log error messages.

        :param iterable msgs: Arguments to be printed.
        """
        # pdb.set_trace()
        self._log_tag(*msgs, tag='ERROR', errorstream=True,
                      report_log_flag=True, highlight=highlight)

    def log_report(self, *msgs, highlight=0):
        """
        Log report messages.

        :param iterable msgs: Arguments to be printed.
        """
        self._log_tag(*msgs, tag='REPORT', report_log_flag=True,
                      highlight=highlight)

    def log_info(self, *msgs, highlight=0):
        """
        Log info messages, printed only if verbose flag is true.

        :param iterable msgs: Arguments to be printed.
        """
        self._log_tag(*msgs, tag='INFO', report_log_flag=True,
                      print_flag=self.verbose, highlight=highlight)

    def log_debug(self, *msgs, highlight=0):
        """
        Log debug messages, printed only if debug flag is true.

        :param iterable msgs: Arguments to be printed.
        """
        self._log_tag(*msgs, tag='DEBUG', errorstream=True,
                      report_log_flag=self.debug,
                      print_flag=self.debug,
                      highlight=highlight)

    def log_full_debug(self, *msgs, highlight=0):
        """
        Log full debug messages, printed only if full debug flag is true.

        :param iterable msgs: Arguments to be printed.
        """
        self._log_tag(*msgs, tag='FULL DEBUG', errorstream=True,
                      print_flag=(self.debug and self.full_debug),
                      highlight=highlight)

    def _get_frame(self):
        return inspect.currentframe().f_back.f_back

    def start(self):
        """
        Start storing the logging in order to flush it into a file.
        """
        frame = self._get_frame()
        # calling_class = frame.f_locals['self'].__class__.__name__
        calling_class = self._get_calling_class_name(frame)
        self.stored_buffer.append('')
        self.threads_log.append('started %d %s:%d %s %s' % (
            len(self.stored_buffer),
            frame.f_code.co_filename,
            frame.f_lineno,
            calling_class,
            str(datetime.datetime.now())))

    def get_timestamp_str(self):
        # tim = datetime.datetime.now()
        tim = self.construct_timestamp
        return 'D%.2d-%.2d_T%.2d-%.2d-%.2d' % (tim.day, tim.month, tim.hour,
                                               tim.minute, tim.second)

    def log_seperator(self):
        output = '\n--------------------------------------------------------\n'
        for idx in range(len(self.stored_buffer)):
            self.stored_buffer[idx] += output
        print(output)

    def log_full_report_into_file(self, output_file, keep_log=False):
        """
        Log all the report of the top-open threads into an output file.
        Additionally, an additional copy is logged into the logs/ directory
        as a backup of the log.

        :param str output_file: Output file path
        :param bool keep_log:
            Flag to decide keeping the log (don't pop the top thread),
            also will print the deepest thread insted of the shallowest.
        """
        frame = self._get_frame()
        # calling_class = frame.f_locals['self'].__class__.__name__
        calling_class = self._get_calling_class_name(frame)
        self.threads_log.append('flushed %d-%d %s:%d %s %s' % (
            len(self.stored_buffer),
            int(keep_log),
            frame.f_code.co_filename,
            frame.f_lineno,
            calling_class,
            str(datetime.datetime.now())))
        directory, file_name, file_ext = extract_file_name(output_file)
        file_ext = '.log'

        self.log_seperator()
        self.log_report('SUMAMRY: Class:', calling_class, 'Timestamp:',
                        datetime.datetime.now())
        file_name = (calling_class + '_' + file_name +
                     self.get_timestamp_str())
        output_file = os.path.join('logs', file_name + file_ext)

        self.log_report('SUMMARY: Logging buffer into %s' % output_file)
        threads_summary = ('\nSUMMARY:' +
                           '\nSUMMARY: '.join(self.threads_log) + '\n')
        if keep_log:
            if not self.stored_buffer:
                return None
            last_stored_buffer = self.stored_buffer[0]
        else:
            last_stored_buffer = self.stored_buffer.pop()
        #FIXME: no extra logs
        #with open_or_create_write_file(output_file) as output:
        #    output.write(cleanstr(last_stored_buffer + threads_summary))
        #    output.close()

        output_file = os.path.join(directory, file_name + file_ext)
        self.log_report('SUMMARY: Logging buffer into %s' % output_file)
        threads_summary = ('\nSUMMARY:' +
                           '\nSUMMARY: '.join(self.threads_log) + '\n')
        self.log_seperator()
        with open_or_create_write_file(output_file) as output:
            output.write(cleanstr(last_stored_buffer + threads_summary))
            output.close()

    def is_debug(self):
        """
        Get the debug flag value

        :rtype: bool
        :returns: debug flag
        """
        return self.debug

    def is_verbose(self):
        """
        Get the verbose flag value

        :rtype: bool
        :returns: verbose flag
        """
        return self.verbose

    def is_full_debug(self):
        """
        Get the full debug flag value

        :rtype: bool
        :returns: full debug flag
        """
        return self.full_debug


#  static logger for the all library
logger = Logger()
