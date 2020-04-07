"""
Module containing a multi-mode visualizer that compares and evaluates
fixing performance, on the three correct, corrupt and fixed texts.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
from constants import (
    DEFAULT_EVALUATOR_ALIGNEMENT, HTML,  # LATEX,
    SMOOTHING, TERMINAL,
    TYPO_ADD, TYPO_DEL, TYPO_NOCHANGE)
from utils.edit_operations import edit_operations
from utils.utils import longest_common_subseq


class MultiViewer:
    """
    Viewer and evaluator of fixed texts vs. correct and corrupt texts
    """
    def __init__(self, alignment=DEFAULT_EVALUATOR_ALIGNEMENT,
                 colored=True):
        self.mode = TERMINAL
        self.alignment = alignment
        self.colored = colored

    # Helper functions for all modes
    def align(self, string, maxlen, mode='center'):
        """
        Align a string to a max length by padding.

        :param str string: Given string
        :param int maxlen: Length of string after alignment
        :param str mode: center, right, left; the mode of alignment
        :rtype: str
        :returns: Aligned string
        """
        if isinstance(string, float):
            string = '%.6f' % string
        rem = maxlen - len(string)
        if rem < 0:
            string = string[:maxlen]
        rem = maxlen - len(string)
        assert rem >= 0, string
        if mode == 'center':
            return (''.join(' ' for _ in range(rem // 2)) +
                    string + ''.join(' ' for _ in range(rem - rem // 2)))
        elif mode == 'right':
            return (''.join(' ' for _ in range(rem)) + string)
        elif mode == 'left':
            return (string + ''.join(' ' for _ in range(rem)))

    def get_brackets_del(self):
        """
        Get brackets/coloring for deletion

        :rtype: pair
        :returns: Pair of closing and opening brackets
        """
        if not self.colored:
            return '', ''
        if self.mode == HTML:
            return '<span style="background-color: #FF0000">', '</span>'
        elif self.mode == TERMINAL:
            return '\033[31m', '\033[0m'

    def get_brackets_del2(self):
        """
        Get brackets/coloring for deletion, 2nd color

        :rtype: pair
        :returns: Pair of closing and opening brackets
        """
        if not self.colored:
            return '', ''
        if self.mode == HTML:
            return '<span style="background-color: #FF00FF">', '</span>'
        elif self.mode == TERMINAL:
            return '\033[32m', '\033[0m'

    def get_brackets_add(self):
        """
        Get brackets/coloring for addition

        :rtype: pair
        :returns: Pair of closing and opening brackets
        """
        if not self.colored:
            return '', ''
        if self.mode == HTML:
            return '<span style="background-color: #00FF00">', '</span>'
        elif self.mode == TERMINAL:
            return '\033[33m', '\033[0m'

    def get_brackets_add2(self):
        """
        Get brackets/coloring for addition, 2nd color

        :rtype: pair
        :returns: Pair of closing and opening brackets
        """
        if not self.colored:
            return '', ''
        if self.mode == HTML:
            return '<span style="background-color: #00FFFF">', '</span>'
        elif self.mode == TERMINAL:
            return '\033[34m', '\033[0m'

    def get_brackets_crossing(self):
        """
        Get brackets/coloring for striking out some text

        :rtype: pair
        :returns: Pair of closing and opening brackets
        """
        if not self.colored:
            return '', ''
        if self.mode == HTML:
            return '<strike>', '</strike>'
        elif self.mode == TERMINAL:
            return '\033[9m', '\033[0m'

    def get_empty_char(self):
        """
        Get filling character

        :rtype: str
        :returns: A string containing empty space
        """
        return ' '

    def get_seperator(self, seperator_length=None):
        """
        Get seperator line between rows

        :param int seperator_length: length of seperator line
        :rtype: str
        :returns: seperator line of length seperator_length or alignment + 2
        """
        if self.mode == HTML:
            return '   '
        elif self.mode == TERMINAL:
            if seperator_length is None:
                return ''.join('_' for _ in range(self.alignment + 2))
            else:
                return ''.join('_' for _ in range(seperator_length + 2))

    def wrap_row_entry(self, text, alignment='center', align_length=None):
        """
        Wrap the cell entry in a row with the proper tags

        :param str text: Given text to be aligned
        :rtype: str
        :returns: wrapped, aligned string
        """
        if self.mode == HTML:
            return '<td align="%s">%s</td>' % (alignment, text)
        elif self.mode == TERMINAL:
            if align_length is None:
                return self.align(text, self.alignment, )
            else:
                return self.align(text, align_length, mode=alignment)

    def wrap_row_header_entry(self, text, align_length=None):
        """
        Wrap the cell entry in a header row with the proper tags

        :param str text: Given text to be aligned
        :rtype: str
        :returns: wrapped, aligned string
        """
        if self.mode == HTML:
            return '<th>%s</th>' % text
        elif self.mode == TERMINAL:
            if align_length is None:
                return self.align(text, self.alignment)
            else:
                return self.align(text, self.alignment)

    def collapse_row(self, corrupt, correct, fixed, new=False, header=False):
        """
        Collapse a triple text comparison into one row

        :param str corrupt: Corrupt text
        :param str correct: Correct text (must be in the middle)
        :param str fixed: Fixed text
        :param bool new: Is a new line (to be marked)
        :param bool header: Header row or not
        :rtype: str
        :returns: Row string description
        """
        if self.mode == HTML:
            if header:
                return '<tr>%s</tr>' % ''.join(map(self.wrap_row_header_entry,
                                                   (corrupt, correct, fixed)))
            else:
                if new:
                    return '<tr style="background-color: #dddddd;">%s</tr>' % (
                        ''.join(map(self.wrap_row_entry,
                                    (corrupt, correct, fixed))))
                else:
                    return '<tr>%s</tr>' % ''.join(
                        map(self.wrap_row_entry,
                            (corrupt, correct, fixed)))
        elif self.mode == TERMINAL:
            if new:
                return ' '.join(('|*|', corrupt, '|*|',
                                 correct, '|*|', fixed, '|'))
            else:
                return ' '.join(('| |', corrupt, '| |',
                                 correct, '| |', fixed, '|'))

    def merge_texts_with_diffs(self, text, diffs):
        """
        Align texts with differences, character by character

        :param str text: Text to be aligned
        :param str diffs: String of fixing operations
        """
        res = []
        for c, cd in zip(text, diffs):
            res.append(c)
            if self.mode == HTML:
                pass
            elif self.mode == TERMINAL:
                res.append(cd)
        return res

    def wrap_page(self, text):
        """
        Wrap page with proper tags

        :param str text: Page to be wrapped
        :rtype: str
        :returns: Wrapped page
        """
        if self.mode == HTML:
            return '<html>\n<body>%s\n<div>\n%s\n</div>\n</body>\n</html>' % (
                self.get_header(), text)
        else:
            return text

    def get_beginning_line(self, seperator):
        """
        Get beginning line at the beginning of the table

        :param str seperator: Seperator used in each column
        :rtype: str
        :returns: String of the start line
        """
        if self.mode == HTML:
            return ''
        elif self.mode == TERMINAL:
            return ' _ %s _ %s _ %s ' % (seperator, seperator, seperator)

    def get_end_line(self, seperator):
        """
        Get ending line at the end of the table

        :param str seperator: Seperator used in each column
        :rtype: str
        :returns: String of the end line
        """
        if self.mode == HTML:
            return ''
        elif self.mode == TERMINAL:
            return '|_|%s|_|%s|_|%s|' % (seperator, seperator, seperator)

    def collapse_comparison_row(self, key, values, R1, R2, header=False):
        if self.mode == HTML:
            if header:
                elements = [self.wrap_row_header_entry(key, align_length=R1)]
                elements += [self.wrap_row_header_entry(v, align_length=R2)
                             for v in values]

                return '<tr>%s</tr>' % ''.join(elements)
            else:
                elements = [self.wrap_row_entry(key, alignment='left',
                                                align_length=R1)]
                elements += [self.wrap_row_entry(self.align(v, R2),
                                                 align_length=R2)
                             for v in values]
                return '<tr>%s</tr>' % ''.join(elements)
        elif self.mode == TERMINAL:
            elements = [self.align(key, R1, mode='left')]
            elements += [self.align(val, R2) for val in values]
            return '|' + '|'.join(elements) + '|'

    def wrap_table_header(self, text):
        """
        Align and wrap the header row of the table with the proper tags

        :param str text: Header content
        :rtype: str
        :returns: wrapped header row
        """
        return self.align(text, self.alignment)
        if self.mode == HTML:
            return '<th>%s</th>' % text
        elif self.mode == TERMINAL:
            return self.align(text, self.alignment)

    def wrap_table(self, text):
        """
        Wrap table with the proper tags

        :param str text: Table content string
        :rtype: str
        :returns: Wrapped table
        """
        if self.mode == HTML:
            return '<table>\n%s\n</table>' % text
        elif self.mode == TERMINAL:
            return text

    def merge_wrapped_pages(self, *pages, mode, seperate=False):
        """
        Merge wrapped pages into a summary page

        :param args(str) pages: full pages contents
        :param int mode: Mode of pages, HTML or TERMINAL
        :param bool seperate: Flag deciding to seperate pages
        :rtype: str
        :return: Merged pages
        """
        self.mode = mode
        result = ''
        first = True
        for page in pages:
            page_unwrapped = ''
            if self.mode == TERMINAL:
                page_unwrapped = page
            elif self.mode == HTML:
                page_unwrapped = '\n'.join(self.get_divs(page))

            if not first and seperate:
                if self.mode == HTML:
                    result += '\n<br><hr><br>\n'
                elif self.mode == TERMINAL:
                    result += self.get_seperator()
            result += page_unwrapped + '\n'
            first = False
        return self.wrap_page(result)

    def add_caption(self, text, caption):
        """
        Add caption/title for the table

        :param str text: Table content
        :param str caption: Caption string
        :rtype: str
        :returns: Table with added captions
        """
        if self.mode == HTML:
            return '<h2>\n%s\n</h2>\n%s\n' % (caption, text)
        elif self.mode == TERMINAL:
            return '%s\n%s\n' % (text, caption)

    def add_metrics(self, metrics):
        """
        Convert metrics to string description

        :param tuple(float) metrics:
            (precision, recall, fscore, precision_del, recall_del, fscore_del,
             precision_add, recall_add, fscore_add)
        :rtype: str
        :returns: string description of metrics values
        """
        if metrics is None:
            return ''
        (precision, recall, fscore, precision_del, recall_del, fscore_del,
            precision_add, recall_add, fscore_add, tp, fp, fn, tp_add, fp_add,
            fn_add, tp_del, fp_del, fn_del, acc) = metrics
        out = '\n'
        if self.mode == TERMINAL:
            out += 'precision: %.7f, recall: %.7f, fscore: %.7f\n' % (
                precision, recall, fscore)
            out += 'To Del precision: %.7f, recall: %.7f, fscore: %.7f\n' % (
                precision_del, recall_del, fscore_del)
            out += 'To Add precision: %.7f, recall: %.7f, fscore: %.7f\n' % (
                precision_add, recall_add, fscore_add)

            out += 'tp: %d, fp: %d, fn: %d\n' % (tp, fp, fn)
            out += 'To Del tp: %d, fp: %d, fn: %d\n' % (tp_del, fp_del, fn_del)
            out += 'To Add tp: %d, fp: %d, fn: %d\n' % (tp_add, fp_add, fn_add)
            out += 'Acc: %d\n' % acc

        elif self.mode == HTML:

            out += 'precision: %.7f, recall: %.7f, fscore: %.7f<br>\n' % (
                precision, recall, fscore)
            out += (
                'To Del precision: %.7f, recall: %.7f, fscore: %.7f<br>\n' %
                (precision_del, recall_del, fscore_del))
            out += (
                'To Add precision: %.7f, recall: %.7f, fscore: %.7f<br>\n' %
                (precision_add, recall_add, fscore_add))

            out += 'tp: %d, fp: %d, fn: %d<br>\n' % (tp, fp, fn)
            out += 'To Del tp: %d, fp: %d, fn: %d<br>\n' % (tp_del, fp_del, fn_del)
            out += 'To Add tp: %d, fp: %d, fn: %d<br>\n' % (tp_add, fp_add, fn_add)
            out += 'Acc: %d<br>\n' % acc

            out = '\n<p>\n%s\n</p>' % out
        return out

    # HTML functions
    def get_divs(self, page, start=0):
        """
        Get divs in a given HTML page, used in merging pages

        :param str page: merged HTML page of divs
        :param int start: Start index to find divs
        :rtype: list(str)
        :returns: pages strings
        """
        unwrapped = []
        while True:
            start = page.find('<div>', start)
            if start == -1:
                break
            start += len('<div>')
            end = page.find('</div>')
            unwrapped.append(page[start: end])
            start = end + len('</div>')
        return unwrapped

    def get_header(self):
        """
        Get HTML/CSS header for HTML pages

        :rtype: str
        :returns: Header HTML code
        """
        if self.mode == HTML:
            return '''
                   <head>
                   <style>
                   table {
                       font-family: arial, sans-serif;
                       border-collapse: collapse;
                       width: 100%;
                   }

                   td, th {
                       border: 1px solid #dddddd;
                       padding: 8px;
                       border: 2px solid black;
                   }

                   </style>
                   </head>
                   '''
        elif self.mode == TERMINAL:
            return ''

    # Functions for all modes
    def get_comparison_table(self, table, columns, R1=26, R2=16):
        """
        Comparison table of the metrics

        :param dict table: Table of metric name to metric values
        :param list(str) columns: Column names of the metric names
        :param int R1: Width of the first column (of model name)
        :param int R2: Width of each of the the remaining columns
        :rtype: str
        :returns: Content of the comparison table
        """
        columns = columns[:9]
        seperator_ln = (R2 + 1) * len(columns) + R1
        res = '\n'
        res += self.get_seperator(seperator_ln) + '\n'
        res += self.collapse_comparison_row('', columns,
                                            R1, R2, header=True) + '\n'
        res += self.get_seperator(seperator_ln) + '\n'
        comparison_keys = list(sorted(table.keys()))
        for comparison_key in comparison_keys:
            values = table[comparison_key][:9]
            res += self.collapse_comparison_row(comparison_key, values,
                                                R1, R2) + '\n'
        res += self.get_seperator(seperator_ln) + '\n'
        return self.wrap_page(self.wrap_table(res))

    def metric_comparison(self, table, metric_names, modes=[]):
        """
        Comparison of metrics

        :param dict table: Table of metric name to metric values
        :param list(str) metric_names: Metric names
        :param list(int) modes: Modes of comparison view, HTML or TERMINAL
        :rtype: str
        :returns: Content of the comparison table
        """
        #print(metric_names)
        #print(table)
        if ((isinstance(modes, list) or isinstance(modes, tuple)) and
                len(modes) > 0):
            results = []
            for mode in modes:
                self.mode = mode
                results.append(self.get_comparison_table(table, metric_names))
            self.mode = TERMINAL

            return tuple(results)
        else:
            self.mode = modes or TERMINAL
            return self.get_comparison_table(table, metric_names)

    def mark_line(self, text, ops):
        """
        Mark and color lines according to the differences description string.

        :param str text: Text to be marked
        :param str ops:
            String with the same length as text showing the difference
            operations
        :rtype: pair(str)
        :returns: Marked text and differences strings
        """
        text_ = ''
        ops_ = ''
        for i in range(len(ops)):
            if ops[i] == '-':
                text_ += self.get_brackets_del()[0]
                ops_ += self.get_brackets_del()[0]
            elif ops[i] == '_':
                text_ += self.get_brackets_del2()[0]
                ops_ += self.get_brackets_del2()[0]
            elif ops[i] == '+':
                text_ += self.get_brackets_add()[0]
                ops_ += self.get_brackets_add()[0]
            elif ops[i] == '*':
                text_ += self.get_brackets_add2()[0]
                ops_ += self.get_brackets_add2()[0]

            if ops[i] == '-' or ops[i] == '_':
                text_ += self.get_brackets_crossing()[0]

            text_ += text[i]
            if ops[i] == '*' or ops[i] == '+':
                ops_ += '+'
            elif ops[i] == '-' or ops[i] == '_':
                ops_ += '-'
            else:
                ops_ += ' '

            if ops[i] == '-' or ops[i] == '_':
                text_ += self.get_brackets_crossing()[1]

            if ops[i] == '-':
                text_ += self.get_brackets_del()[1]
                ops_ += self.get_brackets_del()[1]
            elif ops[i] == '_':
                text_ += self.get_brackets_del2()[1]
                ops_ += self.get_brackets_del2()[1]
            elif ops[i] == '+':
                text_ += self.get_brackets_add()[1]
                ops_ += self.get_brackets_add()[1]
            elif ops[i] == '*':
                text_ += self.get_brackets_add2()[1]
                ops_ += self.get_brackets_add2()[1]

        return text_, ops_

    def split_line_by_length(self, text, ops, length):
        """
        Chunk a long line into a list of lines of fixed length

        :param str text: Given string
        :param ops str: differences string
        :param int length: The fixed length of the text
        :rtype: pair(list(str))
        :returns: pair of splitted texts list and splitted differences list
        """
        texts = []
        ops_res = []
        siz = (len(text) + length - 1) // length
        for idx in range(siz):
            texts.append(text[length * idx:length * (idx + 1)])
            ops_res.append(ops[length * idx:length * (idx + 1)])
        return texts, ops_res

    def diffs_operations(self, text, operations_sequence,
                         comparison_sequence=[]):
        """
        Generate text differences string from the correct text and
        the edit operations to be used.

        :param str text: Correct text
        :param list operations_sequence: List of edit operations
        :param str comparison_sequence:
        :rtype: pair(str)
        :returns:
            Pair of marked corrupt text and corresponding differences string
        """
        text = [c for c in text]
        ops = [self.get_empty_char() for c in text]
        for operation in operations_sequence[::-1]:
            idx = operation[0]
            typ = operation[1]
            if typ == TYPO_DEL:
                ops[idx] = '_' if operation in comparison_sequence else '-'
            elif typ == TYPO_ADD:
                char = operation[2] if operation[2] != '\n' else '|'
                text.insert(idx, char)
                ops.insert(idx,
                           '*' if operation in comparison_sequence else '+')
            else:
                assert typ == TYPO_NOCHANGE
        assert len(ops) == len(text)
        text = ''.join(text)
        ops = ''.join(ops)
        text_res = []
        ops_res = []
        for T, O in zip(*self.split_line_by_length(text, ops, self.alignment)):
            aligned_marked_text, aligned_marked_ops =\
                self.mark_line(self.align(T, self.alignment),
                               self.align(O, self.alignment))
            text_res.append(aligned_marked_text)
            ops_res.append(aligned_marked_ops)
        return text_res, ops_res

    def view_triple_comparison_lines(self, lines, corrupt_opts, nonfixed_opts):
        """
        Construct the triple comparison table from splitted aligned lines of
        correct, corrupt and fixed tests.

        :param list(str) lines: Correct text split into lines of maximal length
        :param list corrupt_opts:
            Edit operations of each line of correct text to corrupt text
        :param list nonfixed_opts:
            Edit operations of each line of correct text to fixed text
        :rtype: str
        :returns: Comparison page aligned by the given lines
        """
        assert len(lines) == len(corrupt_opts) == len(nonfixed_opts)
        seperator = self.get_seperator()
        correct_rows = [seperator]
        fixed_rows = [seperator]
        corrupt_rows = [seperator]
        corrupt_rows.append(self.wrap_table_header('corrupt text'))
        fixed_rows.append(self.wrap_table_header('fixed text'))
        correct_rows.append(self.wrap_table_header('correct text'))
        corrupt_rows.append(seperator)
        fixed_rows.append(seperator)
        correct_rows.append(seperator)

        idxs = [len(correct_rows)]
        for correct, corrupt_op, nonfix_op in zip(lines, corrupt_opts,
                                                  nonfixed_opts):
            corrupt, corrupt_diffs = self.diffs_operations(correct, corrupt_op,
                                                           nonfix_op)
            fixed, fixed_diffs = self.diffs_operations(correct, nonfix_op,
                                                       corrupt_op)
            correct, correct_diffs = self.diffs_operations(correct, [], [])
            max_len = max(len(correct), len(fixed), len(corrupt))
            while len(corrupt) < max_len:
                corrupt.append(self.align('', self.alignment))
                corrupt_diffs.append(self.align('', self.alignment))
            while len(fixed) < max_len:
                fixed.append(self.align('', self.alignment))
                fixed_diffs.append(self.align('', self.alignment))
            while len(correct) < max_len:
                correct.append(self.align('', self.alignment))
                correct_diffs.append(self.align('', self.alignment))
            correct_rows.extend(self.merge_texts_with_diffs(correct,
                                                            correct_diffs))
            fixed_rows.extend(self.merge_texts_with_diffs(fixed, fixed_diffs))
            corrupt_rows.extend(self.merge_texts_with_diffs(corrupt,
                                                            corrupt_diffs))
            idxs.append(len(correct_rows))

        corrupt_rows.append(seperator)
        fixed_rows.append(seperator)
        correct_rows.append(seperator)
        out = '\n'
        for idx, corrupt, correct, fixed in zip(range(len(correct_rows)),
                                                corrupt_rows, correct_rows,
                                                fixed_rows):
            if corrupt == correct == fixed == seperator:
                if idx < idxs[0]:
                    out += self.get_beginning_line(seperator)
                    out += '\n'
                elif idx >= idxs[-1]:
                    out += self.get_end_line(seperator)
                    out += '\n'
                else:
                    assert False
                    out += '| |%s| |%s| |%s|' % (seperator, seperator,
                                                 seperator)
                    out += '\n'
            elif idx in idxs:
                out += self.collapse_row(corrupt, correct, fixed, new=True)
                out += '\n'
            else:
                is_header = idx < idxs[0]
                out += self.collapse_row(corrupt, correct, fixed,
                                         header=is_header)
                out += '\n'
        out += '\n'
        out = self.wrap_table(out)
        return out

    def view_triple_comparison(self, correct, corrupt, fixed,
                               modes=[], metrics=None, caption=''):
        """
        Get triple comparisons for all the modes of viewing

        :param str correct: Correct text
        :param str corrupt: Corrupt text
        :param str fixed: Fixed text
        :param list(int) modes: List of modes
        :param dict metrics: Metric values dictionary
        :param str caption: Caption string
        :rtype: str or tuple(str)
        :returns:
            Full comparison pages in all given modes,
            or only TERMINAL by default if no modes are given
        """
        corruption_operations = edit_operations(correct, corrupt)
        nonfixed_operations = edit_operations(correct, fixed)
        if ((isinstance(modes, list) or isinstance(modes, tuple)) and
                len(modes) > 0):
            results = []
            for mode in modes:
                self.mode = mode
                results.append(self.view_triple_comparison_ops(
                    correct, corruption_operations, nonfixed_operations,
                    metrics, caption))
            self.mode = TERMINAL

            return tuple(results)
        else:
            self.mode = modes or TERMINAL
            return self.view_triple_comparison_ops(
                correct, corruption_operations, nonfixed_operations,
                metrics, caption)

    def view_triple_comparison_ops(self, correct, corruption_operations,
                                   nonfixed_operations, metrics, caption):
        """
        View full triple comparison of correct text and edit operations to
        corrupt file and fixed file, in addition to a caption and the
        comparison metrics.

        :param str correct: Correct text
        :param list corruption_operations:
            List of edit operations from correct to corrupt text
        :param list nonfixed_operations:
            List of edit operations from correct to fixed text
        :param dict metrics: Dictionary of metrics evaluations
        :param str caption: Caption to be added to comparison
        :rtype: str
        :returns: Full comparison page
        """

        # non_fixed_corruptions = corruption_operations & nonfixed_operations
        # fixed_corruptions = corruption_operations - non_fixed_corruptions
        # wrongly_fixed_correct = nonfixed_operations - non_fixed_corruptions
        # all_edit_operations = corruption_operations | non_fixed_corruptions
        lines, corrupt_opts, nonfixed_opts = [], [], []

        offset, corrup_idx, nonfix_idx = -1, 0, 0
        corrupt_opts_line, nonfixed_opts_line = [], []
        for idx in range(len(correct)):
            while corrup_idx < len(corruption_operations):
                if corruption_operations[corrup_idx][0] > idx:
                    break
                op = corruption_operations[corrup_idx]
                op = (op[0] - offset - 1,) + op[1:]
                if (correct[corruption_operations[corrup_idx][0]] != '\n' or
                        corruption_operations[corrup_idx][1] != TYPO_DEL):
                    corrupt_opts_line.append(op)
                corrup_idx += 1
            while nonfix_idx < len(nonfixed_operations):
                if nonfixed_operations[nonfix_idx][0] > idx:
                    break
                op = nonfixed_operations[nonfix_idx]
                op = (op[0] - offset - 1,) + op[1:]
                if (correct[nonfixed_operations[nonfix_idx][0]] != '\n' or
                        nonfixed_operations[nonfix_idx][1] != TYPO_DEL):
                    nonfixed_opts_line.append(op)
                nonfix_idx += 1

            if correct[idx] == '\n':
                lines.append(correct[offset + 1: idx])
                corrupt_opts.append(corrupt_opts_line)
                nonfixed_opts.append(nonfixed_opts_line)
                corrupt_opts_line = []
                nonfixed_opts_line = []
                offset = idx

        lines.append(correct[offset + 1:])
        corrupt_opts.append(corrupt_opts_line)
        nonfixed_opts.append(nonfixed_opts_line)
        corrupt_opts_line = []
        nonfixed_opts_line = []
        offset = idx
        assert '\n'.join(lines) == correct
        comparison = self.view_triple_comparison_lines(lines, corrupt_opts,
                                                       nonfixed_opts)
        comparison = comparison + self.add_metrics(metrics)
        comparison = self.add_caption(comparison, caption)
        return self.wrap_page(comparison)

    def num_operations(self, corruption_operations, nonfixed_operations):
        """
        Evaluate operations using fscore, precision and recall

        :param list corruption_operations: True positives
        :param list nonfixed_operations: predicted negatives
        :rtype: triple(float)
        :returns: precision, recall, fscore

        >>> Q = MultiViewer()
        >>> ','.join(
        ...     ('%.3f' % q
        ...      for q in Q.num_operations([1, 2, 3, 5], [2, 4, 5, 8, 9])))
        '0.400,0.500,0.444'
        """
        non_fixed_corruptions = longest_common_subseq(corruption_operations,
                                                      nonfixed_operations)
        fixed_corruptions = len(corruption_operations) - non_fixed_corruptions
        wrongly_fixed_correct = (len(nonfixed_operations) -
                                 non_fixed_corruptions)
        non_fixed_corruptions, fixed_corruptions, wrongly_fixed_correct
        # Positive <==> predicting it's an error (hence fixing it)
        tp, fp, fn = (fixed_corruptions, wrongly_fixed_correct,
                      non_fixed_corruptions)

        precision = tp / float(tp + fp + SMOOTHING)
        recall = tp / float(tp + fn + SMOOTHING)
        fscore = 2 * recall * precision / (recall + precision + SMOOTHING)
        if tp == fp == fn == 0:
            precision = recall = fscore = 1.0
        return precision, recall, fscore, tp, fp, fn

    def evaluate(self, correct, corrupt, fixed, modes=[], caption=''):
        """
        Evaluate a given correct, corrupt and fixed texts.

        :param str correct: Correct text
        :param str correct: Corrupt text
        :param str correct: Fixed text
        :param list(int) modes: Modes of comparison views
        :param str caption: Caption to be added to comparisons
        :rtype: pair(tuple)
        :returns:
            Pair of the evaluation metrics tuple and a tuple of comparison
            descriptions.
        """
        corruption_operations = edit_operations(correct, corrupt)
        corruption_operations_add = [op for op in corruption_operations
                                     if op[1] == TYPO_ADD]
        corruption_operations_del = [op for op in corruption_operations
                                     if op[1] == TYPO_DEL]
        nonfixed_operations = edit_operations(correct, fixed)

        nonfixed_operations_add = [op for op in nonfixed_operations
                                   if op[1] == TYPO_ADD]
        nonfixed_operations_del = [op for op in nonfixed_operations
                                   if op[1] == TYPO_DEL]

        precision, recall, fscore, tp, fp, fn = self.num_operations(
            corruption_operations, nonfixed_operations)
        precision_add, recall_add, fscore_add, tp_add, fp_add, fn_add =\
            self.num_operations(corruption_operations_add,
                                nonfixed_operations_add)
        precision_del, recall_del, fscore_del, tp_del, fp_del, fn_del =\
            self.num_operations(corruption_operations_del,
                                nonfixed_operations_del)
        acc = 1 - int(bool(nonfixed_operations))
        metrics = (precision, recall, fscore, precision_del, recall_del,
                   fscore_del, precision_add, recall_add, fscore_add,
                   tp, fp, fn, tp_add, fp_add, fn_add, tp_del, fp_del, fn_del, acc)
        # out = view_triple_comparison(correct, corrupt, fixed)
        out = self.view_triple_comparison(correct, corrupt, fixed,
                                          modes=modes, metrics=metrics,
                                          caption=caption)
        '''
        out += 'precision: %.7f, recall: %.7f, fscore: %.7f\n' % (
                    precision, recall, fscore)
        out += 'To Del precision: %.7f, recall: %.7f, fscore: %.7f\n' % (
                    precision_del, recall_del, fscore_del)
        out += 'To Add precision: %.7f, recall: %.7f, fscore: %.7f\n' % (
                    precision_add, recall_add, fscore_add)
        '''
        if not isinstance(out, tuple):
            out = (out, )
        return (metrics,) + out
        return (precision, recall, fscore, precision_del, recall_del,
                fscore_del, precision_add, recall_add, fscore_add), out
