import getpass
import multiprocessing
import os
import socket
import _pickle as pickle
# from os.path import join


USERNAME = getpass.getuser()
HOSTNAME = socket.gethostname()

# Choose which virtual runner is used
WHARFER = 1
RISEML = 2
VIRTUAL_RUNNER = None  # WHARFER  # None or RISEML


# DEFAULT_GRAPHS_DUMP_DIR = '/output/'
#DEFAULT_DATA_LOAD_DIR = '/Users/thrax/thesis_benchmarks/benchmarks/'
#DEFAULT_DATA_LOAD_DIR = '/local/hdd/exports/data/matthias-hertel/'

#DEFAULT_MODEL_LOAD_DIR = '/nfs/students/mostafa-mohamed/paper/dumps'
#DEFAULT_DATA_LOAD_DIR = '/nfs/students/mostafa-mohamed/paper'

DEFAULT_ROOT_DIR = '.'
DEFAULT_MODEL_DUMP_DIR = 'dumps'
DEFAULT_MODEL_LOAD_DIR = 'dumps'
DEFAULT_DATA_LOAD_DIR = 'benchmarks_root'
DEFAULT_BENCHMARK_DUMP_DIR = 'benchmark_dumps'

if os.path.isdir('/nfs/students/mostafa-mohamed/paper_v2'):
    DEFAULT_ROOT_DIR = '/nfs/students/mostafa-mohamed/paper_v2'
    DEFAULT_MODEL_DUMP_DIR = '/nfs/students/mostafa-mohamed/paper_v2/dumps'
    DEFAULT_MODEL_LOAD_DIR = '/nfs/students/mostafa-mohamed/paper_v2/dumps'
    DEFAULT_DATA_LOAD_DIR = '/nfs/students/matthias-hertel'
    DEFAULT_BENCHMARK_DUMP_DIR = '/nfs/students/mostafa-mohamed/paper_v2/benchmark_dumps'

try:
    from polyaxon_client.tracking import get_outputs_path
    print('Polyaxon output path:', get_outputs_path())
    DEFAULT_DATA_LOAD_DIR = '/data/1/amin/benchmarks/'
    DEFAULT_DATA_LOAD_DIR = '/data/1/matthias-hertel/'
    DEFAULT_MODEL_LOAD_DIR = '/data/1/amin/dumps/'
    DEFAULT_MODEL_DUMP_DIR = os.path.join(get_outputs_path(), 'dumps')
    DEFAULT_BENCHMARK_DUMP_DIR = os.path.join(get_outputs_path(), 'benchmark_dumps')
except Exception as err:
    pass
    # print(err)


# Hyperparameters
HISTORY_LENGTH = 20 # Even number
SAMPLE_VOCAB_SIZE = 0  # 30000

#  Constants
# CACHE = True
BATCH_SIZE = 256#4096 * 2
NUM_THREADS = max(min(8, multiprocessing.cpu_count()), multiprocessing.cpu_count() - 1)
DEFAULT_RANDOM_SEED = 41

# constants
SAMPLE_WORDS = ['system', 'american', 'island', 'people']
IDENTITY_INPUT_FORMAT = 'I'
CORRECT_INPUT_FOMRAT = 'C'
EDIT_CORRECT_CORRUPT_INPUT_FOMRAT = 'EGC'
DEFAULT_EVALUATOR_ALIGNEMENT = 56
HTML = 'HTML'
TERMINAL = 'TERMINAL'
LATEX = 'LATEX'
VIEWER3D_MODES = (HTML, TERMINAL, LATEX)
MICRO_EPS = 1e-31
EPS = 1e-7
SMOOTHING = 1e-7
INFINITY = 10 ** 20
NEG_INFINITY = -INFINITY
CORRUPTION_SPACE_SPLIT = 0
CORRUPTION_HYPHEN_SPLIT = 1
CORRUPTION_TYPO = 2
CORRUPTION_MERGE_NEXT = 3
ACTIONS_DIM = 10
LABELS_DIM = 6

SPLIT_ENUM = -2
USE_WHOLE_ENUM = -1

TYPO_DEL = -1
TYPO_ADD = 1
TYPO_CHANGE = 2
TYPO_NOCHANGE = 0
CORRUPTION_CHOICES = (CORRUPTION_SPACE_SPLIT,
                      CORRUPTION_HYPHEN_SPLIT,
                      CORRUPTION_TYPO,
                      CORRUPTION_MERGE_NEXT,
                      )
CORRUPTION_CHOICES_CHANCES = dict(zip(CORRUPTION_CHOICES, (4, 1, 2, 4)))


BACKWARD = 'backward'
FORWARD = 'forward'

SPECIAL = "()[]{}!?,;:.\"-#$%&*+/<=>@\\^_`|~"

#  Flags
DEBUG = True
VERBOSE = True
FULL_DEBUG = False


with open('encoder.pkl', 'rb') as fl:
    ENCODER_DICT = pickle.load(fl)
UNK = ENCODER_DICT['UNK']
SOS = ENCODER_DICT['SOS']
EOS = ENCODER_DICT['EOS']
DECODER_DICT = [k for _, k in sorted([(v, k) for k, v in ENCODER_DICT.items()])]
__SPECIAL = [21, 22, 26, 39, 45, 46, 47, 66, 67, 69, 70, 71, 75, 77, 78, 80,
            82, 83, 84, 85, 86, 87, 89, 90, 92, 95, 98, 99, 102, 103,
            107, 110, 111, 118, 120, 131, 134, 137, 143, 147, 149, 172, 173,
            186, 191, 193, 197]
ALPHABET = [v for k, v in enumerate(DECODER_DICT) if k not in __SPECIAL and k < 200 and k != 0]
SPECIAL = [v for k, v in enumerate(DECODER_DICT) if k in __SPECIAL and k < 200]


def charset_of(c):
    if c in ALPHABET: return 1
    if c in SPECIAL: return 2
    return -1


def encode(string):
    return [ENCODER_DICT.get(c, UNK) for c in string]


def decode(lst, exclude_pads=True):
    return ''.join((DECODER_DICT[x] for x in lst))


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
        return 2 + encode(op_char)[0]
    else:
        assert False, 'no valid operation' + str((op_typ, op_char))

# ENums

class Enum:
    def __init__(self, **kwargs):
        self.merge(kwargs)

    def keys(self):
        return list(self.__dict__.keys())

    def merge(self, kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)


FIXERS_ENUM = Enum(bicontext_fixer='bicontext_fixer', dp_fixer='dp_fixer', e2e_fixer='e2e_fixer')
DATASETS_ENUM = Enum(wikipedia='wikipedia', simple_wikipedia='SimpleWikipedia')
E2E_MODES_ENUM = Enum(decision='decision_e2e', language='language_e2e',
                      full_e2e='full_e2e')
MODELS_ENUM = Enum(forward_language_model='forward_LM',
                   backward_language_model='backward_LM',
                   e2e_model='e2e_model')
BENCHMARKS_ENUM = [
    '0_0.1', '0_0.2',
    '0_0.3', '0_0.4',
    '0_0.5', '0_0.6',
    '0_0.7', '0_0.8',
    '0_0.9', '0_1',
    '0_inf',
    '0.2_0.1', '0.2_0.2',
    '0.2_0.3', '0.2_0.4',
    '0.2_0.5', '0.2_0.6',
    '0.2_0.7', '0.2_0.8',
    '0.2_0.9', '0.2_1',
    '0.2_inf',
    '0.1_0.1', '0.1_0.2',
    '0.1_0.3', '0.1_0.4',
    '0.1_0.5', '0.1_0.6',
    '0.1_0.7', '0.1_0.8',
    '0.1_0.9', '0.1_1',
    '0.1_inf',
]


DEFAULT_BENCHMARK = BENCHMARKS_ENUM[int(os.environ.get('TOKENIZATION_BENCHMARK', 0))]
DEFAULT_DATASET = DATASETS_ENUM.wikipedia  #simple_wikipedia

print('DATA:', USERNAME, HOSTNAME, DEFAULT_MODEL_DUMP_DIR, DEFAULT_MODEL_LOAD_DIR,
      DEFAULT_DATA_LOAD_DIR, DEFAULT_BENCHMARK_DUMP_DIR, DEFAULT_BENCHMARK,
      str(BENCHMARKS_ENUM.index(DEFAULT_BENCHMARK) + 1) + '/' + str(len(BENCHMARKS_ENUM)))
