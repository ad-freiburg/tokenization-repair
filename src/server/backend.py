from typing import Tuple

from urllib.parse import unquote

from src.helper.files import read_file
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.spelling.spelling_beam_search_corrector import SpellingBeamSearchCorrector


FOLDER = "html/"
HTML_PATH = FOLDER + "repair.html"
FILES = ["style.css"]
QUERY_PREFIX = "repair?query="
FWD_MODEL_NAME = "fwd1024"
ROBUST_FWD_MODEL_NAME = "fwd1024_noise0.2"


MODES = [("tr", "Tokenization repair without spelling errors"),
         ("trn", "Tokenization repair with spelling errors"),
         ("spell", "Spelling correction")]


def mode_select_html(selected_mode: str):
    html = """<select name="mode">"""
    for mode, label in MODES:
        if mode == selected_mode:
            selected_str = " selected=\"selected\""
        else:
            selected_str = ""
        html += """<option value="%s"%s>%s</option>""" % (mode, selected_str, label)
    html += "</select>"
    return html


def get_token_repairer(model: UnidirectionalLMEstimator) -> BatchedBeamSearchCorrector:
    return BatchedBeamSearchCorrector(model,
                                      insertion_penalty=0,
                                      deletion_penalty=0,
                                      n_beams=3,
                                      verbose=False)


def get_spelling_corrector(model: UnidirectionalLMEstimator) -> SpellingBeamSearchCorrector:
    return SpellingBeamSearchCorrector(model,
                                       n_beams=100,
                                       branching_factor=100,
                                       consecutive_insertions=2,
                                       char_penalty=6,
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


def load_fwd_model(model_name: str) -> UnidirectionalLMEstimator:
    model = UnidirectionalLMEstimator()
    model.load(model_name)
    return model


class Backend:
    def __init__(self):
        self.html = read_file(HTML_PATH)
        self.file_contents = {file: read_file(FOLDER + file) for file in FILES}
        fwd_model = load_fwd_model(FWD_MODEL_NAME)
        self.token_repairer = get_token_repairer(fwd_model)
        robust_fwd_model = load_fwd_model(ROBUST_FWD_MODEL_NAME)
        self.robust_token_repairer = get_token_repairer(robust_fwd_model)
        self.spelling_corrector = get_spelling_corrector(fwd_model)

    def predict(self, query: str, mode: str):
        if mode == "tr":
            return self.token_repairer.correct(query)
        elif mode == "trn":
            return self.robust_token_repairer.correct(query)
        elif mode == "spell":
            predicted, segmentation = self.spelling_corrector.correct(query)
            return predicted
        else:
            return ""

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
        select_html = mode_select_html(mode)
        html = self.html % (query, select_html, answer)
        return html
