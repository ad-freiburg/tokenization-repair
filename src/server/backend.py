from typing import Tuple, Optional

from urllib.parse import unquote

from src.helper.files import read_file
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.corrector.labeling.labeling_corrector import LabelingCorrector
from src.corrector.load.labeling import load_labeling_corrector
from src.spelling.spelling_beam_search_corrector import SpellingBeamSearchCorrector
from src.corrector.load.model import load_unidirectional_model, load_bidirectional_model
from src.corrector.load.beam_search import load_two_pass_corrector, INF


FOLDER = "html/"
HTML_PATH = FOLDER + "repair.html"
FILES = ["style.css"]
QUERY_PREFIX = "repair?query="
MAX_QUERY_LENGTH = 512


MODES = [("unidir", "UNI"),
         ("unidir-r", "UNI+"),
         ("bidir", "BID"),
         ("bidir-r", "BID+"),
         ("the-one", "BID+ the one"),
         ("spell", "spelling correction")]
DEFAULT_MODE = "the-one"


def mode_select_html(selected_mode: str):
    html = """<select name="mode">"""
    for mode, label in MODES:
        if mode == selected_mode or (selected_mode == "" and mode == DEFAULT_MODE):
            selected_str = " selected=\"selected\""
        else:
            selected_str = ""
        html += """<option value="%s"%s>%s</option>""" % (mode, selected_str, label)
    html += "</select>"
    return html


def get_bs_token_repairer(model: UnidirectionalLMEstimator,
                          bidirectional_model: Optional[BidirectionalLabelingEstimator] = None) \
        -> BatchedBeamSearchCorrector:
    return BatchedBeamSearchCorrector(model,
                                      insertion_penalty=0,
                                      deletion_penalty=0,
                                      n_beams=5,
                                      verbose=False,
                                      labeling_model=bidirectional_model)


def get_labeling_token_repairer(model: BidirectionalLabelingEstimator,
                                robust: bool) -> LabelingCorrector:
    return load_labeling_corrector(robust=robust, typos=robust, p=INF, model=model)


def get_spelling_corrector(model: UnidirectionalLMEstimator) -> SpellingBeamSearchCorrector:
    return SpellingBeamSearchCorrector(model,
                                       n_beams=100,
                                       branching_factor=100,
                                       consecutive_insertions=2,
                                       char_penalty=8,
                                       space_penalty=0,
                                       max_edits_per_word=1,
                                       verbose=False)


def is_query(request: str) -> bool:
    return request.startswith(QUERY_PREFIX)


def get_query(request: str) -> Tuple[str, str]:
    query = request[len(QUERY_PREFIX):]
    split_vals = query.split('&')
    mode = split_vals[-1][len("mode="):]
    query = '&'.join(split_vals[:-1])
    query = query.replace("+", " ")
    query = unquote(query)
    return query, mode


def html_safe_value(value: str) -> str:
    value = value.replace("\"", "&quot;")
    return value


def wrap_result(result: str) -> str:
    wrapped = "<b>Result:</b><br>\n" + \
        "<table>\n" + \
        "<tr><td>%s</td></tr>" + \
        "</table>\n"
    wrapped = wrapped % result
    return wrapped


class Backend:
    def __init__(self):
        self.html = read_file(HTML_PATH)
        self.file_contents = {file: read_file(FOLDER + file) for file in FILES}
        fwd = load_unidirectional_model(backward=False, robust=False)
        fwd_robust = load_unidirectional_model(backward=False, robust=True)
        bidir = load_bidirectional_model(robust=False)
        bidir_robust = load_bidirectional_model(robust=True)
        self.unidir = get_bs_token_repairer(fwd)
        self.unidir_robust = get_bs_token_repairer(fwd_robust)
        self.bidir = get_bs_token_repairer(fwd, bidir)
        self.bidir_robust = get_bs_token_repairer(fwd_robust, bidir_robust)
        self.the_one = get_bs_token_repairer(fwd_robust, bidir_robust)
        self.the_one.insertion_penalty = -6.32
        self.the_one.deletion_penalty = -6.8
        self.spelling_corrector = get_spelling_corrector(fwd)
        self.correctors = {
            "unidir": self.unidir,
            "unidir-r": self.unidir_robust,
            "bidir": self.bidir,
            "bidir-r": self.bidir_robust,
            "the-one": self.the_one,
            "spell": self.spelling_corrector
        }

    def predict(self, query: str, mode: str):
        if mode not in self.correctors:
            return ""
        query = query[:MAX_QUERY_LENGTH]
        if mode == "spell":
            predicted, segmentation = self.spelling_corrector.correct(query)
        else:
            predicted = self.correctors[mode].correct(query)
        return predicted

    def answer(self, request: str) -> str:
        if request in FILES:
            return self.file_contents[request]
        if is_query(request):
            query, mode = get_query(request)
            answer = self.predict(query, mode)
        else:
            query = answer = mode = ""
        query = html_safe_value(query)
        answer = html_safe_value(answer)
        if len(answer) > 0:
            answer = wrap_result(answer)
        select_html = mode_select_html(mode)
        html = self.html % (query, select_html, answer)
        return html
