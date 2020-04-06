"""
Module containing a set of utility functions and small independent algorithms.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import _pickle
import errno
import os
import random
import re
import sys
import threading
from os import listdir
from os.path import isdir, isfile, join

from constants import MICRO_EPS, SMOOTHING, TYPO_NOCHANGE, TYPO_DEL, TYPO_ADD
from .edit_operations import detailed_edit_operations

import numpy as np


def take_first_n(gen, n):
    for idx, item in enumerate(gen):
        if idx >= n:
            break
        yield item
    return


def take_n_samples_per_line(generator, context_length, samples_per_line,
                            backward):
    for data in generator:
        if len(data) > context_length:
            if backward:
                idxs = np.random.randint(context_length,
                                         len(data),
                                         size=samples_per_line)
            else:
                idxs = np.random.randint(context_length + context_length - 1,
                                         context_length + len(data),
                                         size=samples_per_line)
        else:
            idxs = np.random.randint(context_length, context_length + len(data),
                                     size=samples_per_line)
        if backward:
            data_ = np.append(data, np.zeros(context_length))
        else:
            data_ = np.append(np.zeros(context_length), data)
        for idx in idxs:
            yield data_[idx - context_length: idx + 1]
    return


def get_shuffled_lines(path, newlns=None, shuffle=True):
    with open(path, 'rb') as fil:
        mp = np.array(ALPHABET_MAP)
        if newlns is None:
            newlns = [0] + np.cumsum([len(ln) for ln in fil]).tolist()

        while True:
            idxs = list(range(len(newlns) - 1))
            if shuffle:
                np.random.shuffle(idxs)
            # if num_lines_per_epoch is not None:
            #     idxs = idxs[:num_lines_per_epoch]
            for idx in idxs:
                fil.seek(newlns[idx])
                data = fil.readline()
                data_ = np.frombuffer(data, dtype=np.uint8, count=len(data))

                #### DEBUG
                """
                from configs import get_language_model_config
                config = get_language_model_config(inference=True)
                from models.mostafa.character_model import CharacterModel
                char_model = CharacterModel(config)

                print(char_model.codes_to_str(ALPHABET_MAP[data_]))
                print(data.decode('utf8'))
                assert char_model.codes_to_str(ALPHABET_MAP[data_]) ==\
                    data.decode('utf8'), (
                        char_model.codes_to_str(ALPHABET_MAP[data_]),
                        data.decode('utf8'), ALPHABET_MAP[data_])
                print('\n\n')
                """
                #### DEBUG

                yield ALPHABET_MAP[data_]
    return


def read_shuffled_lines(path, idxs, shuffle=True):
    with open(path, 'rb') as fil:
        newlns = [0] + np.cumsum([len(ln) for ln in fil]).tolist()
        for idx in idxs:
            fil.seek(newlns[idx])
            line = cleanstr(fil.readline().decode('utf8'))
            yield line
    return


def encode_operation(op):
    if isinstance(op, tuple):
        op_typ, op_char = op
    else:
        op_typ = op
    if op_typ == TYPO_DEL:
        return 0
    elif op_typ == TYPO_NOCHANGE:
        return 1
    elif op_typ == TYPO_ADD:
        # print("ENC!!", op_char)
        return 2 + ALPHABET_MAP[op_char]  # .char_code(op_char)
    else:
        assert False, 'no valid operation' + str((op_typ, op_char))


def ls_recursive(path, with_arg=True, endswith=None):
    args = []
    for arg in os.listdir(path):
        full_arg = os.path.join(path, arg)
        if os.path.isdir(full_arg):
            for xarg in ls_recursive(full_arg):
                args.extend(full_arg)
        elif os.path.isfile(full_arg):
            if endswith is None or full_arg.endswith(endswith):
                args.append(full_arg)
    return args


def edits_generator(path_correct, path_corrupt, context_length, shuffle=False):
    from .logger import logger
    logger.log_debug(path_correct, path_corrupt)
    with open(path_correct, 'rb') as fil_correct:
        with open(path_corrupt, 'rb') as fil_corrupt:
            newlns_correct = [0] + np.cumsum([len(ln) for ln in fil_correct]).tolist()
            newlns_corrupt = [0] + np.cumsum([len(ln) for ln in fil_corrupt]).tolist()

            assert len(newlns_correct) == len(newlns_corrupt)
            idxs = list(range(len(newlns_corrupt) - 1))
            while True:
                if shuffle:
                    np.random.shuffle(idxs)
                for idx in idxs:
                    fil_correct.seek(newlns_correct[idx])
                    fil_corrupt.seek(newlns_corrupt[idx])
                    correct = fil_correct.readline()
                    correct_data = ALPHABET_MAP[np.frombuffer(
                        correct, dtype=np.uint8, count=len(correct))]

                    corrupt = fil_corrupt.readline()
                    corrupt_data = ALPHABET_MAP[np.frombuffer(
                        corrupt, dtype=np.uint8, count=len(corrupt))]

                    editops = detailed_edit_operations(corrupt, correct)
                    # logger.log_debug(corrupt.decode(), highlight=6)
                    # logger.log_debug(correct.decode(), highlight=7)
                    for idx_corrupt, idx_correct, action in editops:
                        before = correct_data[idx_correct - context_length: idx_correct]
                        after = corrupt_data[idx_corrupt + 1: idx_corrupt + context_length+ 1]
                        current = corrupt_data[idx_corrupt: idx_corrupt + 1]
                        action_code = np.array([encode_operation(action)])
                        if len(before) < context_length:
                            before = np.append(np.zeros(
                                context_length - len(before), dtype=np.uint8),
                                before)
                        if len(after) < context_length:
                            after = np.append(after, np.zeros(
                                context_length - len(after), dtype=np.uint8))
                        ###########
                        """
                        from configs import get_language_model_config
                        config = get_language_model_config(inference=True)
                        from models.mostafa.character_model import CharacterModel
                        char_model = CharacterModel(config)

                        if action_code[0] != 1:#not in [0, 1]:
                            logger.log_debug(
                                char_model.codes_to_str(before), ' <> ',
                                char_model.codes_to_str(current), ' <> ',
                                char_model.codes_to_str(after),
                                'DEL' if action_code[0] == 0 else 'NOP' if action_code[0] == 1 else
                                "ADD '" + (EMPTY_CHAR + CHAR_SET + TOKENIZATION_DELIMITERS)[action_code[0] - 2] + "'",
                                highlight=4)
                        # logger.log_debug(np.shape(before), np.shape(current),
                        #         np.shape(after), np.shape(action_code))
                        """
                        ###########
                        yield before, current, after, action_code
    return


def context_generator(path, context_length, samples_per_line, shuffle=True,
                      backward=False):
    with open(path, 'rb') as fil:
        newlns = [0] + np.cumsum([len(ln) for ln in fil]).tolist()
        total_num_lines = len(newlns)
    generator = get_shuffled_lines(path, newlns, shuffle=shuffle)
    return total_num_lines, take_n_samples_per_line(generator, context_length,
                                                    samples_per_line, backward)


def cleanstr(ss):
    """
    Clean string from non-ascii characters

    :param str ss: Given input string
    :rtype: str
    :returns: Cleaned string
    """
    return ''.join(filter(lambda x: ord(x) in range(128), ss))


def get_corresponding_file(fil, prefix, newext=None):
    """
    Get corresponding file with some subdirectory

    :param str fil: File path
    :param str prefix: prefix of the subdirectory
    :param str newext:
        New extension of corresponding file, if None it won't be changed
    :rtype: str
    :returns: Corresponding path

    >>> get_corresponding_file('./home/world/here/text_file.pdf', 'corex')
    './home/world/here/corex/text_file.pdf'
    >>> get_corresponding_file('./home/world/here/text_file.pdf', 'corex/word')
    './home/world/here/corex/word/text_file.pdf'
    >>> get_corresponding_file('./home/world/here/text_file.pdf', 'corex',
    ...                        newext='edits')
    './home/world/here/corex/text_file.edits'
    """
    path, file_name, ext = extract_file_name(fil)
    if newext is None:
        newext = ext
    return os.path.join(path, prefix, file_name + '.' + newext)


def get_corresponding_corrupt_file(fil, delimiters_only=False):
    """
    Get corresponding corrupt file

    :param str fil: File path
    :param bool delimiters_only: Corresponding file with no typos or with typos
    :rtype: str
    :returns: Corresponding corrupt file path

    >>> get_corresponding_corrupt_file('./home/world/here/text_file.pdf')
    './home/world/here/corrupt_typos/text_file.pdf'
    >>> get_corresponding_corrupt_file('./home/world/here/text_file.pdf', True)
    './home/world/here/corrupt/text_file.pdf'
    """
    if delimiters_only:
        return get_corresponding_file(fil, 'corrupt')
    else:
        return get_corresponding_file(fil, 'corrupt_typos')


def get_corresponding_edit_file(fil, delimiters_only=False):
    """
    Get corresponding edit file

    :param str fil: File path
    :param bool delimiters_only: Corresponding file with no typos or with typos
    :rtype: str
    :returns: Corresponding edit file path

    >>> get_corresponding_edit_file('./home/world/here/text_file.pdf')
    './home/world/here/corrupt_typos/edits/text_file.edits'
    >>> get_corresponding_edit_file('./home/world/here/text_file.pdf', True)
    './home/world/here/corrupt/edits/text_file.edits'
    """
    if delimiters_only:
        return get_corresponding_file(fil, 'corrupt/edits', 'edits')
    else:
        return get_corresponding_file(fil, 'corrupt_typos/edits', 'edits')


def beam_search(initial_state, is_terminal_state, get_next_state,
                comparison_key, beam_size):
    """
    Beam search for the closest target, using a queue of fixed size.

    :param int beam_size: queue size
    :param State initial_state: Initial state of beam search
    :param function is_terminal_state:
        Function that checks if a state is a terminal state
    :param function get_next_state:
        Function that gets all the next states of some state
    :param function comparison_key:
        Function that is used to compare states
    :rtype: State
    :returns: Final state
    """
    queue = [initial_state]
    results = []
    while queue:
        final_states = [state for state in queue if is_terminal_state(state)]
        results.extend(final_states)
        queue = sorted([next_state for state in queue
                        if not is_terminal_state(state)
                        for next_state in get_next_state(state)],
                       key=comparison_key)
        # queue = queue[-beam_size: ]
        queue = queue[: beam_size]
    results = sorted(results, key=comparison_key)
    return results[0]


def sort_files_by_size(files):
    """
    Sort a list of file names ascendingly according to their size.

    :param list(str) files: Given files
    :rtype: list(str)
    :returns: sorted list of files

    >>> sort_files_by_size(['setup.cfg', '__init__.py', 'Makefile'])
    ['setup.cfg', '__init__.py', 'Makefile']
    >>> sort_files_by_size(['Makefile', 'setup.cfg', '__init__.py'])
    ['setup.cfg', '__init__.py', 'Makefile']
    """
    return [fil for _, fil in sorted(
        [(os.stat(fil).st_size, fil) for fil in files])]


def chunk_files_by_square_size(tups, MAX=10**7):
    files_siz = sorted([((len(tup[0]) + 1) * (len(tup[1]) + 1), tup)
                        for tup in tups])[::-1]
    idx = 0
    while idx < len(files_siz):
        new_chunk = []
        tot = 0
        while idx < len(files_siz):
            siz, fil = files_siz[idx]
            if not new_chunk:
                new_chunk.append(fil)
                tot += siz
            else:
                if tot + siz < MAX:
                    new_chunk.append(fil)
                    tot += siz
                else:
                    break
            idx += 1
        yield new_chunk


def rechunk_files_uniformly_by_size(files, chunksize):
    res = [None for _ in files]
    sorted_files = sort_files_by_size(files)
    visited = [0 for _ in files]
    tot = 0
    for r in range(chunksize):
        st, en, step = r, len(files), chunksize
        # if r % 2 == 1:
        #    st, en, step = ((len(files) - r - 1) // chunksize * chunksize + r,
        #                    -1, -chunksize)
        for i in range(st, en, step):
            res[i] = sorted_files[-tot]
            visited[-tot] += 1
            tot += 1
    assert all(x == 1 for x in visited)
    assert sort_files_by_size(res) == sorted_files
    return res


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes,
                                                           bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def get_vocab(*texts):
    """
    Extract all the words from a given text.

    :param str text: The given text.
    :rtype: set(str)
    :returns: A set of the words in the text

    >>> sorted(list(get_vocab('Hello world')))
    ['Hello', 'world']
    >>> sorted(list(get_vocab(
    ...    'Hello world I am- hello world bored')))
    ['Hello', 'I', 'am', 'bored', 'hello', 'world']
    """
    vocab = {}
    for text in texts:
        for line in text.split('\n'):
            rsearch = re.search(r"(\w[\w']*)\s?(.*)", line)
            while rsearch:
                word, line = rsearch.groups()
                rsearch = re.search(r"(\w[\w']*)\s?(.*)", line)
                # word = word.lower()
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    return vocab


def flatten(lists):
    '''
    Flatten list of lists

    :param list lists: list of lists
    :rtype: list
    :returns: flattened list

    >>> flatten([[1, 2], [4, 6]])
    [1, 2, 4, 6]
    '''
    res = []
    for ls in lists:
        if isinstance(ls, np.ndarray):
            print('ndarry\n', ls)
            print('list\n', ls.tolist())
            print()
            res.extend(ls.tolist())
        else:
            res.extend(ls)
    return res


def data_desc(dataset_dict, remove_iter=False):
    """
    Construct short represntation of dataset/configuration dictionaries

    :param dict dataset_dict: Information dictionary
    :param bool remove_iter: Exclude epochs in representation
    :rtype: str
    :returns: short description
    """
    info = []
    if not remove_iter:
        if 'iter' in dataset_dict.keys():
            info.append('it{}'.format(dataset_dict['iter']))
    if 'dataset' in dataset_dict.keys():
        info.append('{}'.format(dataset_dict['dataset']))
    if 'architecture' in dataset_dict.keys():
        info.append('Arch{}'.format(dataset_dict['architecture']))
    if 'perturbate' in dataset_dict.keys():
        info.append('P{}'.format(dataset_dict['perturbate']))
    if 'reverse_input' in dataset_dict.keys():
        if isinstance(dataset_dict['reverse_input'], tuple):
            info.append(
                '+'.join(('R' if s else 'N'
                          for s in dataset_dict['reverse_input'])))
        else:
            info.append('R' if dataset_dict['reverse_input'] else 'N')
    if 'return_seqs' in dataset_dict.keys():
        if dataset_dict['return_seqs']:
            info.append('A')
    if 'fold_num' in dataset_dict.keys():
        info.append('F{}out{}'.format(dataset_dict['fold_num'],
                                      dataset_dict['num_folds']))
    if 'files_to_take' in dataset_dict.keys():
        info.append('tot{}'.format(dataset_dict['files_to_take']))
    if 'delimiters_only' in dataset_dict.keys():
        info.append('T' if dataset_dict['delimiters_only'] else 'X')
    if ('default_weights' in dataset_dict.keys() and
            dataset_dict['default_weights']):
        info.append('DW')
    if 'beamsize' in dataset_dict.keys():
        # TODO: Remove condition in future, but needs to rename all tuner dumps
        if dataset_dict['beamsize'] != 2:
            info.append('B%d' % dataset_dict['beamsize'])
    if ('lookforward' in dataset_dict.keys() and
            not dataset_dict['lookforward']):
        info.append('NLF')
    return '-'.join(info)


class ThreadsafeGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.thread_lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.thread_lock:
            yield self.generator.__next__()


def threadsafe_iterator(generator):
    """
    Threadsafe version of a given generator
    """
    def threadsafe(*args, **kwargs):
        return ThreadsafeGenerator(generator(*args, **kwargs))
    return threadsafe


def gen_chunker(gen, size):
    cont = True
    idx = 0
    while cont:
        objects = []
        for _ in range(size):
            # print('populating element', idx)
            idx += 1
            try:
                objects.append(next(gen))
            except StopIteration:
                cont = False
                break
        yield objects
    return


def chunker(seq, size):
    """
    Chunk a given sequence to chunks of fixed size.

    :param list seq: Given list of items to be chunked
    :param int size: Chunks size
    :rtype: Generator
    :returns: Generator of sublists(chunks)
    """
    # From Stackoverflow
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))


def read_all(paths, pickle=False):
    # paths = sorted(paths)
    res = []
    for path in paths:
        if pickle:
            with open(path, 'rb') as fl:
                res.append(_pickle.load(fl))
        else:
            with open(path, 'r') as fl:
                res.append(fl.read())
    return res


def open_or_create_write_file(filename, permissions='w'):
    """
    Open file and create corresponding directories.

    :param str filename: file path
    :param str permissions: file permissions:
    :rtype: File
    :returns: opened file
    """
    if (os.path.dirname(filename) and
            not os.path.exists(os.path.dirname(filename))):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return open(filename, permissions)


def makedirs(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


def get_onlyfiles(path):
    """
    Get all files in a given directory.

    :param str path: The path of the directory.
    :rtype: list(str)
    :returns: List of full paths of files.
    """
    return [join(path, f) for f in listdir(path) if isfile(join(path, f)) or
            (f.upper() == f and isdir(join(path, f)))]


def extract_file_name(full_path):
    """
    Extract the directory and file name from a full path.

    :param str full_path: The full path of the file.
    :rtype: triple
    :returns: tuple containing directory, file_name, file_extension

    >>> extract_file_name('./home/world/here/text_file.pdf')
    ('./home/world/here', 'text_file', 'pdf')
    >>> extract_file_name('./home/world/here/text_file')
    ('./home/world/here', 'text_file', '')
    """
    folders = full_path.split('/')
    file_exts = folders[-1].split('.')

    directory = str.join('/', folders[:-1])
    if len(file_exts) > 1:
        file_name = str.join('.', file_exts[:-1])
        file_ext = file_exts[-1]
    else:
        file_name = '.'.join(file_exts)
        file_ext = ''

    return directory, file_name, file_ext


def append_prefix_to_file_name(full_path, prefix):
    """
    Append a prefix to the file in the given full path.

    :param str full_path: The full path of the given file.
    :param str prefix: The prefix to be added to the file name
    :rtype: str
    :returns: the new file path

    >>> append_prefix_to_file_name('./home/world/here/text_file.pdf', 'hello_')
    './home/world/here/hello_text_file.pdf'
    """
    directory, file_name, file_ext = extract_file_name(full_path)
    return join(directory, prefix + file_name + '.' + file_ext)


def get_all_files(path, ext='.txt'):
    """
    Get all files in a given directory (or 1 file), that has
    a specific extension.

    :param str path: The path of the file/directory.
    :param str ext: The extension of the files.
    :rtype: list(str)
    :returns: List of full paths of files.
    """
    res = []
    if os.path.isdir(path):
        for fil in get_onlyfiles(path):
            res.extend(get_all_files(fil, ext))
        return res
    elif os.path.isfile(path) and os.stat(path).st_size > 0:
        if path.endswith(ext):
            res.append(path)
    return res


def get_all_files_from_args(args, ext='.txt', include_arg=True):
    """
    Get all files with some extension from a set of directories.

    :param list(str) args: List of directories
    :param str ext: Extension of the searched file
    :rtype: list(str)
    :returns: List of returned files
    """
    res = []
    for arg in args:
        files = get_all_files(arg, ext)
        if include_arg:
            files = [os.path.join(arg, fil) for fil in files]
        res.extend(files)
    return res


def biased_choice(prs):
    """
    Make a biased random choice given a list of probabilities

    :param list[float] prs: List of probabilities, not necessary normalized
    :rtype: int
    :returns: Index of chosen element
    """
    tot = sum(prs)
    p = random.random()
    for idx, pr in enumerate(prs):
        if pr > p * tot:
            return idx
        p -= pr / float(tot)
    return len(prs)


def findmax(iterable, default=None):
    """
    Find the first maximum of an iterable object and the index of this element.

    :param iterable iterable: The iterable object
    :param default: The default value of the maximum.
    :returns: A tuple (object, index)
    :rtype: tuple

    >>> findmax([1, 2, 66, 3, 66])
    (66, 2)
    >>> findmax([1, 2, 66, 3, 66], 66)
    (66, 2)
    >>> findmax([1, 2, 66, 3, 66], 100)
    (100, -1)
    >>> findmax([1])
    (1, 0)
    >>> findmax([])
    (None, -1)
    >>> findmax([], 0)
    (0, -1)
    """
    res = (default, -1)
    for idx, x in enumerate(iterable):
        if res[0] is None or x > res[0] or (x == res[0] and res[1] == -1):
            res = (x, idx)
    return res


def mean(iterable):
    """
    Compute the average of a gien iterable

    :param iterable iterable: The generator of the given data.
    :rtype: float
    :returns: The average value of the iterable

    >>> mean([1, 2, 3])
    2.0
    >>> mean([])
    0.0
    """
    if iterable:
        total, siz = 0, 0
        for x in iterable:
            total += x
            siz += 1
        if siz:
            return total / float(siz)
    return 0.0


def compute_relative_log_improvement(baseline, improved, epsilon=0.01):
    """
    Compute the relative log improvement of the baseline and improved

    :param ndarray baseline: Baseline metrics
    :param ndarray improved: Improved metrics
    :rtype: ndarray
    :returns: Imporvement measure

    >>> '%.2f' % compute_relative_log_improvement(0.5, 0.9)
    '0.36'
    >>> '%.2f,%.2f' % compute_relative_log_improvement((0.5, 0.6), (0.9, 0.8))
    '0.36,0.25'
    """
    inf = np.log(epsilon)
    improvement = (np.log(improved) - np.log(baseline)) / (epsilon - inf)
    improvement = np.sign(improvement) * np.sqrt(np.abs(improvement))
    if improvement.shape:
        improvement = tuple(improvement.tolist())
        return improvement
    return improvement


def compute_relative_improvement(baseline, improved):
    """
    Compute the relative improvement of the baseline and improved

    :param ndarray baseline: Baseline metrics
    :param ndarray improved: Improved metrics
    :rtype: ndarray
    :returns: Imporvement measure

    >>> '%.2f,%.2f' % compute_relative_improvement((0.5, 0.6), (0.9, 0.8))
    '0.80,0.50'
    """
    res = 1.0 - (1.0 - np.array(improved)) / (1.0 - np.array(baseline) + 1e-9)
    res = tuple(res.tolist())
    return res


def longest_increasing_subseq(sequence, cmp=None):
    """
    Longest increasing subsequence

    :param iterable sequence: The given sequence
    :param function cmp: Comparison function
    :rtype: list
    :returns: Longest increasing subsequence

    >>> longest_increasing_subseq([1, 4, 2, 10, 7, 101])
    [1, 4, 10, 101]
    >>> longest_increasing_subseq([1, 4, 3, 10, 1, 7], lambda x, y: x > y)
    [4, 3, 1]
    """
    F = [1 for i in range(len(sequence))]
    prev = [-1 for j in range(len(sequence))]
    for i in range(len(sequence)):
        for j in range(i):
            if F[j] + 1 > F[i]:
                if ((cmp is None and sequence[j] <= sequence[i]) or
                        (cmp is not None and cmp(sequence[j], sequence[i]))):
                    F[i] = F[j] + 1
                    prev[i] = j
    last = findmax(F)[1]
    res = []
    while last >= 0:
        res.append(sequence[last])
        last = prev[last]
    return list(reversed(res))


def longest_common_subseq(X, Y):
    """
    Compute longest common subsequence between to strings.

    :param str X: first string
    :param str X: second string
    :rtype: int
    :returns: The length of the longest common subsequence

    >>> longest_common_subseq('AbCCxyz', 'ACyyy')
    3
    >>> longest_common_subseq('abc', 'xyz')
    0
    >>> longest_common_subseq('', 'x')
    0
    >>> longest_common_subseq('x', '')
    0
    >>> longest_common_subseq('', '')
    0
    """
    rows = [[0 for j in range(len(Y) + 1)] for i in range(2)]
    for i in range(1, len(X) + 1):
        rows[i & 1] = [0 for j in range(len(Y) + 1)]
        for j in range(1, len(Y) + 1):
            if X[i - 1] == Y[j - 1]:
                rows[i & 1][j] = rows[(i - 1) & 1][j - 1] + 1
            else:
                rows[i & 1][j] = max(rows[(i - 1) & 1][j], rows[i & 1][j - 1])
    return rows[len(X) & 1][len(Y)]


def edit_dist(X, Y):
    """
    Compute the edit distance between 2 strings using add, delete or substitute

    :param str X: first string
    :param str Y: second string
    :rtype: int
    :returns: The edit distance between the 2 strings

    >>> edit_dist('abc', 'ab')
    1
    >>> edit_dist('abchelloworld', 'helloword')
    4
    >>> edit_dist('xyz', '')
    3
    """
    rows = [[len(X) + len(Y) + 1 for j in range(len(Y) + 1)] for i in range(2)]
    for i in range(0, len(X) + 1):
        rows[i & 1] = [len(X) + len(Y) + 1 for j in range(len(Y) + 1)]
        for j in range(0, len(Y) + 1):
            if i == 0 and j == 0:
                rows[i & 1][j] = 0
            if i > 0:
                rows[i & 1][j] = min(rows[i & 1][j], rows[(i - 1) & 1][j] + 1)
            if j > 0:
                rows[i & 1][j] = min(rows[i & 1][j], rows[i & 1][j - 1] + 1)
            if i > 0 and j > 0 and X[i - 1] != Y[j - 1]:
                rows[i & 1][j] = min(rows[i & 1][j],
                                     rows[(i - 1) & 1][j - 1] + 1)
            if i > 0 and j > 0 and X[i - 1] == Y[j - 1]:
                rows[i & 1][j] = rows[(i - 1) & 1][j - 1]

    return rows[len(X) & 1][len(Y)]


def edit_dist_traceback(X, Y, delimiters=' ', visualize=False):
    """
    Compute the edit distance between 2 strings using add, delete or substitute

    :param str X: first string
    :param str Y: second string
    :rtype: int
    :returns: The edit distance between the 2 strings

    >>> edit_dist_traceback('correct words are here', 'correct words are here')
    (0, 0)
    >>> edit_dist_traceback('correct words are here', 'correctwords are her e')
    (1, 1)
    >>> edit_dist_traceback('correct words', 'correctw ords')
    (1, 1)
    >>> edit_dist_traceback('correct words', 'correctwords')
    (1, 0)
    >>> edit_dist_traceback('correct words', 'correct w or ds')
    (0, 2)
    >>> edit_dist_traceback('co rrect words', 'correct w or ds',
    ...                     visualize=True)
    (1, 2, [2], [9, 12])
    """
    TYPO_NOCHANGE, TYPO_ADD, TYPO_DEL = 0, 1, 2
    MAX = len(X) + len(Y) + 1
    rows = [[MAX for j in range(len(Y) + 1)] for i in range(len(X) + 1)]
    space_cntX = [[MAX for j in range(len(Y) + 1)] for i in range(len(X) + 1)]
    space_cntY = [[MAX for j in range(len(Y) + 1)] for i in range(len(X) + 1)]
    traceback = [[None for j in range(len(Y) + 1)] for i in range(len(X) + 1)]
    for i in range(0, len(X) + 1):
        rows[i] = [len(X) + len(Y) + 1 for j in range(len(Y) + 1)]
        for j in range(0, len(Y) + 1):
            if i == 0 and j == 0:
                rows[i][j] = 0
                space_cntX[i][j] = 0
                space_cntY[i][j] = 0
            if i > 0:
                if rows[i][j] >= rows[i - 1][j] + 1:
                    space_cntY[i][j] = space_cntY[i - 1][j]
                    space_cntX[i][j] = space_cntX[i - 1][j]
                    if X[i - 1] in delimiters:
                        space_cntX[i][j] += 1
                    rows[i][j] = rows[i - 1][j] + 1
                    traceback[i][j] = TYPO_ADD
            if j > 0:
                if rows[i][j] >= rows[i][j - 1] + 1:
                    space_cntX[i][j] = space_cntX[i][j - 1]
                    space_cntY[i][j] = space_cntY[i][j - 1]
                    if Y[j - 1] in delimiters:
                        space_cntY[i][j] += 1
                    rows[i][j] = rows[i][j - 1] + 1
                    traceback[i][j] = TYPO_DEL
            if i > 0 and j > 0 and X[i - 1] == Y[j - 1]:
                space_cntX[i][j] = space_cntX[i - 1][j - 1]
                space_cntY[i][j] = space_cntY[i - 1][j - 1]
                rows[i][j] = rows[i - 1][j - 1]
                traceback[i][j] = TYPO_NOCHANGE

    if visualize:
        x, y = len(X), len(Y)
        Rx = []
        Ry = []
        while traceback[x][y] is not None:
            step = traceback[x][y]
            if step == TYPO_ADD:
                if X[x - 1] in delimiters:
                    Rx.append(x - 1)
                x -= 1
            elif step == TYPO_DEL:
                if Y[y - 1] in delimiters:
                    Ry.append(y - 1)
                y -= 1
            elif step == TYPO_NOCHANGE:
                x -= 1
                y -= 1
            else:
                assert False
        Rx = Rx[::-1]
        Ry = Ry[::-1]
        return space_cntX[len(X)][len(Y)], space_cntY[len(X)][len(Y)], Rx, Ry

    return space_cntX[len(X)][len(Y)], space_cntY[len(X)][len(Y)]


def one_hot(array, classes):
    if isinstance(array, (int, np.int)) or array.ndim == 0:
        out = np.zeros((classes,), dtype=np.int8)
        out[array] = 1
        return out
    else:
        array = np.array(array, dtype=np.int32)
        idxs = []
        for i, sz in enumerate(array.shape):
            msk = np.ogrid[:sz]
            shape = np.ones(array.ndim, dtype=np.int32)
            shape[i] = sz
            idxs.append(msk.reshape(shape))

        idxs.append(array)
        out = np.zeros(array.shape + (classes,), dtype=np.uint8)
        out[tuple(idxs)] = 1
        return out


def distort_distribution(preds, temperature=0.5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + MICRO_EPS) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return preds


def multinomial(preds, rolls=2):
    probas = np.random.multinomial(rolls, preds, 1)
    probas = probas / np.sum(probas)
    probas = probas[0, :]
    return probas


def normalize_distribution(preds, temperature=0.5, rolls=2):
    """
    Normalize a given probability vector

    :param list[float] preds: Probability vector
    :param int rolls: number of rolls in normalization
    :param float temperature: Temperature of normalization
    :rtype: list[float]
    :returns: List of normalized probabilities
    """
    preds = distort_distribution(preds, temperature)
    if rolls > 0:
        return multinomial(preds, rolls)
    return preds


def compute_precision(actual, predicted):
    """
    Compute the precision of predictions.

    :param list actual: Actual/correct values
    :param list actual: predicted values
    :rtype: float
    :returns: The precision of correctly predicted output
    """
    return (sum(int(x == y) for x, y in zip(actual, predicted)) /
            float(min(len(actual), len(predicted))))


def get_edit_evaluation(expected, actual, delimiters=' ', func=None):
    """
    Compute evaluation metrics: recall, precision, f1score

    :param list expected: Expected to be predicted sequence values (correct)
    :param list actual: Actually predicted sequence values (predicted by model)
    :param function func:
        A function that can be applied on the sequences. If None, then identity
    :rtype: triple(float)
    :returns: recall, precision, f1score

    >>> '%.3lf,%.3lf,%.3lf' % get_edit_evaluation(
    ...     'correctfix ation here ar ebad',
    ...     'correct fixation here are b ad')
    '0.500,0.400,0.444'
    >>> '%.3lf,%.3lf,%.3lf' % get_edit_evaluation(
    ...        'cor rectfix ation here ar eok',
    ...        'cor rect fixati on here are ok')
    '0.600,0.500,0.545'
    >>> '%.3lf,%.3lf,%.3lf' % get_edit_evaluation('#rsfx#o#n',
    ...                                       '#rs#fx#on#',
    ...                                        delimiters='#')
    '0.667,0.500,0.571'
    >>> '%.3lf,%.3lf,%.3lf' % get_edit_evaluation('#rsfx#on#he#ar#eb',
    ...                                       '#rs#fx#on#he#are#b',
    ...                                        delimiters='#')
    '0.800,0.667,0.727'
    """
    if func is not None:
        expected = str(map(func, expected))
        actual = str(map(func, actual))
    false_neg, false_pos = edit_dist_traceback(expected, actual,
                                               delimiters=delimiters)
    positives = sum(1 for char in expected if char in delimiters)
    true_pos = positives - false_neg
    recall = true_pos / (true_pos + false_neg + SMOOTHING)
    precision = true_pos / (true_pos + false_pos + SMOOTHING)
    f1score = recall * precision * 2.0 / (recall + precision + SMOOTHING)
    return recall, precision, f1score


def get_evaluation(expected, actual, func=None):
    """
    Compute evaluation metrics: recall, precision, f1score

    :param list expected: Expected to be predicted sequence values (correct)
    :param list actual: Actually predicted sequence values (predicted by model)
    :param function func:
        A function that can be applied on the sequences. If None, then identity
    :rtype: triple(float)
    :returns: recall, precision, f1score

    >>> '%.3lf,%.3lf,%.3lf' % get_evaluation([2, 3, 5, 7, 10, 1],
    ...                                      [1, 2, 5, 7, 8],)
    '0.500,0.600,0.545'
    """
    if func is None:
        true_positives = longest_common_subseq(actual, expected)
    else:
        true_positives = longest_common_subseq(list(map(func, actual)),
                                               list(map(func, expected)))
    recall, precision = (true_positives / float(len(expected) + SMOOTHING),
                         true_positives / float(len(actual) + SMOOTHING))
    f1score = recall * precision * 2.0 / (recall + precision + SMOOTHING)
    return recall, precision, f1score


def concatenate_dict_values(dicts, axis=None):
    keys = list(dicts[0].keys())
    res = {key: np.concatenate([d[key] for d in dicts], axis=axis)
           for key in keys}
    return res


def stack_dict_values(dicts, axis=None):
    keys = list(dicts[0].keys())
    res = {key: np.stack([d[key] for d in dicts], axis=axis)
           for key in keys}
    return res


def dict_values_chunker(dic, size):
    len_seq = len(list(dic.values())[0])
    return ({key: np.array(dic[key][pos: pos + size], dtype=np.int) for key in dic.keys()}
            for pos in range(0, len_seq, size))
